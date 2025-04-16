# import h5py
# import shap
# import torch
# import shap
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from Models import our as mil
# from scipy import interpolate
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import h5py
import sys, argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import torch


from PIL import Image
from matplotlib import cm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
# Load color map (using 'jet' colormap for heatmap visualization)
colormap = plt.cm.get_cmap('jet')
parser = argparse.ArgumentParser(description='Train our model')
parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate [0.0002]')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
parser.add_argument('--dataset', default='LUAD_TMB_2_1_1', type=str, help='Dataset folder name')
# parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
parser.add_argument('--model', default='our', type=str, help='model our')
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
parser.add_argument('--average', type=bool, default=True,
                    help='Average the score of max-pooling and bag aggregating')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--agg', type=str, help='which agg')
parser.add_argument('--c_path', nargs='+',
                    default=None, type=str,
                    help='directory to confounders')  # './datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
parser.add_argument('--dir', type=str,help='directory to save logs')
parser.add_argument('--folder_path', default='./multi_graph_1',
                        help='number of GNN message passing layers (default: 5)')

args = parser.parse_args()

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores
def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset
# Load color map (using 'jet' colormap for heatmap visualization)
colormap = plt.get_cmap('jet')
def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key]
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def _get_stride(coordinates: np.ndarray) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)

    return stride


def _MIL_heatmap_for_slide(
    coords: np.ndarray, scores: np.ndarray, colormap=None
) -> np.ndarray:
    """
    Args:
        coords: Coordinates for each patch in the WSI
        scores: Scores for each patch
        colormap: Colormap to visualize the score contribution
    Returns:
        Heatmap as an RGB numpy array
    """
    stride = 512  # Can be adjusted to suit the actual image size
    scaled_map_coords = coords // stride

    # Make a mask, 1 where coordinates have attention, 0 otherwise
    mask = np.zeros(scaled_map_coords.max(0) + 1)
    for coord in scaled_map_coords:
        mask[coord[0], coord[1]] = 1

    grid_x, grid_y = np.mgrid[
        0 : scaled_map_coords[:, 0].max() + 1, 0 : scaled_map_coords[:, 1].max() + 1
    ]

    if scores.ndim < 2:
        scores = np.expand_dims(scores, 1)
    activations = interpolate.griddata(scaled_map_coords, scores, (grid_x, grid_y))
    activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

    # Normalize activations to range [0, 1]
    activations_min = activations.min()
    activations_max = activations.max()
    activations_normalized = (activations - activations_min) / (activations_max - activations_min + 1e-8)

    # Apply colormap to normalized activations
    heatmap = (colormap(activations_normalized[..., 0])[:, :, :3] * 255).astype(np.uint8)

    return heatmap


# filename = "1548631-6"  #"1485428-5-HE"#"TCGA-49-6767-01Z-00-DX1.53459c0e-b8ec-4893-9910-87b63c503134"#"1411193-4-HE"#"1434028-2-HE"#"1430851-3-HE" #"1434028-2-HE"
# 加载数据
# feats_TME = 'P:\\lung_cancer\\TME\\{filename}_graph.h5'.format(filename=filename)
# # feats_TME = 'P:\\lung_cancer\\TME_luad\\{filename}_graph.h5'.format(filename=filename)
# with h5py.File(feats_TME, 'r') as hf:
#     node_features = hf['node_features'][:]
#     edges = hf['edges'][:]
feats_TME = [os.path.abspath(os.path.join(args.folder_path, f)) for f in os.listdir(args.folder_path) if
                 os.path.isfile(os.path.join(args.folder_path, f))]
# feats_csv_path1 = './datas/xiangya2/all_graph/{filename}.h5'.format(filename=filename)
# feats_csv_path1 = 'P:\\lung_cancer\\LUAD_feature\\multi_graph_1\\{filename}.h5'.format(filename=filename.split('.')[0])
with h5py.File(feats_TME[0], 'r') as hf:
    x_img_256 = hf['x_img_256'][:]
    x_img_256_edge = hf['x_img_256_edge'][:]
    x_img_512 = hf['x_img_512'][:]
    x_img_512_edge = hf['x_img_512_edge'][:]
    # x_img_1024 = hf['x_img_1024'][:]
    # x_img_1024_edge = hf['x_img_1024_edge'][:]
    node_features = hf['node_features'][:]

    # 读取 edges
    edges = hf['edges'][:]

x_img_256 = torch.tensor(x_img_256).to(device)
x_img_512 = torch.tensor(x_img_512).to(device)
# x_img_1024 = torch.tensor(x_img_1024).to(device)
x_img_256_edge = torch.tensor(x_img_256_edge).to(device)
x_img_512_edge = torch.tensor(x_img_512_edge).to(device)
# x_img_1024_edge = torch.tensor(x_img_1024_edge).to(device)
node_features = torch.tensor(node_features).to(device)
edges = torch.tensor(edges).to(device)

import Models.our3 as mil
milnet = mil.fusion_model_graph(args = args, in_channels=args.feats_size, hidden_channels=args.hidden_channels, out_channels =args.num_classes).to(device)
model = milnet.to(device)
# td = torch.load(r'M:\project_P53\lung_cancer\baseline_TMB_2\xiangya2_TMB_2_1_1_our_None_fulltune\0\1_2.pth')
#xiangya2使用our2中的fusion_model
# td = torch.load(r'P:\lung_cancer\test_models\TMB\1_4.pth')#our2
# td = torch.load(r'P:\lung_cancer\test_models\ALK\1_0.pth')#our2
td = torch.load(r'./test_models/3_4.pth')#our2



model.load_state_dict(td, strict=False)
model.eval()

# 获取模型输出
x_256_score,x_512_score, TME_fea = model(x_img_256, x_img_512, x_img_256_edge, x_img_512_edge, node_features.to(torch.float32), edges)
scores = x_256_score.detach().cpu().numpy()


####################
filename = [os.path.splitext(f)[0] for f in os.listdir('./input') if os.path.isfile(os.path.join('./input', f))]
filename = filename[0]
##读取位置信息
# 加载数据
feats= './features/FEATURES_DIRECTORY_256/h5_files/{filename}.h5'.format(filename=filename)
with h5py.File(feats, 'r') as hf:
    node_features = hf['features'][:]
    coords = hf['coords'][:]



# 读取 WSI 图像坐标
# feats_TME = 'M:\\project_P53\\lung_cancer\\features\\256\\RESULTS_DIRECTORY\\patches\\{filename}.h5'.format(filename=filename)
# with h5py.File(feats_TME, 'r') as hf:
#     coords = hf['coords'][:]

from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches

heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': 1, 'blur': False, 'custom_downsample': 1}
vis_patch_size = (256, 256)
slide_path = './input/{filename}.svs'.format(filename=filename) ##1285271-7-HE
mask_file = './heatmap/{filename}_mask.pkl'.format(filename=filename)
seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False} #,'keep_ids': ' ', 'exclude_ids':' '
filter_params = {'a_t':1, 'a_h':1, 'max_n_holes':2}


# seg_params = load_params(process_stack.loc[i], seg_params)
# filter_params = load_params(process_stack.loc[i], filter_params)
# vis_params = load_params(process_stack.loc[i], vis_params)

import os
wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
sample = {'k': 100, 'mode': 'topk', 'name': 'topk_high_attention', 'sample': True, 'seed': 1}
sample_save_dir =  './results/topk_high_attention/256/{filename}_1'.format(filename=filename)   #os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
os.makedirs(sample_save_dir, exist_ok=True)
print('sampling {}'.format(sample['name']))
sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
    patch = wsi_object.wsi.read_region(tuple(s_coord), 0, (256, 256)).convert('RGB')
    dpi = 600
    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, filename, s_coord[0], s_coord[1], s_score)), dpi=(dpi, dpi))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['blue', 'yellow'])#'jet'
heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,
						          cmap=cmap, alpha=0.4, **heatmap_vis_args,
						          binarize=False,
						  		  blank_canvas=False,
						  		  thresh=-1,  patch_size = vis_patch_size,
						  		  overlap=0.5,
						  		  top_left=None, bot_right = None)

heatmap.save('./heatmap/{filename}_256.jpg'.format(filename=filename), quality=100)

