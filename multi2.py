import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
import copy
import random
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from modules.utils import device, setcolor_mesh_batched, load_state_dict, find_red_green_pixels, show_mask, show_points, loadmesh, add_spheres, write_ply
from modules.dataset_multi import DecoderDataset
from modules.render import save_renders, Renderer
from modules.click_attention import ClickAttention
from modules.decoder import Decoder


def train_decoder(args):
    print('train decoder')
    num_gpus = torch.cuda.device_count()
    print('number of available gpus: %d' % num_gpus)

    # Constrain most sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    render_high = Renderer(dim=(args.render_res, args.render_res))

    # MLP Settings: predicting 3D probability mask
    attention = ClickAttention(256, args).to(device)
    mlp = Decoder(depth=args.depth, width=[512] + [256] * args.depth, out_dim=2, input_dim=512,
                  positional_encoding=args.positional_encoding,
                  sigma=args.sigma).to(device)

    parameters = list(attention.parameters()) + list(mlp.parameters())
    optim = torch.optim.Adam(parameters, args.learning_rate)

    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        attention = torch.nn.DataParallel(attention, device_ids=device_ids_list)
        mlp = torch.nn.DataParallel(mlp, device_ids=device_ids_list)

    # Create datasets, meshes, and features for multiple objects
    datasets = []
    all_indices_list = []
    for i in range(len(args.decoder_data_dirs)):
        dataset = DecoderDataset(args.decoder_data_dirs[i], args.meshes[i], args)
        datasets.append(dataset)
        all_indices_list.append(list(range(len(dataset))))
        print(f'Length of dataset {i + 1}: {len(dataset)}')

    # Load learned 3D features
    features = load_features(args)

    print ("load 3D")

    # Initialize variables
    start_epoch = 0
    start_batch_indices = [0] * len(datasets)
    save_path = os.path.join(args.save_dir, args.model_name)
    losses = []
    print ("initialize")

    if args.continue_train:
        checkpoint = torch.load(save_path, map_location=device)
        attention.load_state_dict(checkpoint['attention_state_dict'])
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch_indices = checkpoint['batch_indices']
        all_indices_list = checkpoint['shuffled_indices_list']
        losses = checkpoint['losses']
        print ("continue")

    for epoch in range(start_epoch, args.num_epochs):
        batch_indices = start_batch_indices.copy()
        # Shuffle indices at each epoch start
        for idx, all_indices in enumerate(all_indices_list):
            if epoch != start_epoch or start_batch_indices[idx] == 0:
                random.shuffle(all_indices)
        print ("start")

        # Create samplers and data loaders for each dataset
        samplers = []
        data_loaders = []
        num_batches_list = []
        for i in range(len(datasets)):
            start_batch = batch_indices[i]
            sampler = SubsetRandomSampler(all_indices_list[i][start_batch * args.batch_size:])
            samplers.append(sampler)
            data_loader = DataLoader(datasets[i], sampler=sampler, batch_size=args.batch_size)
            data_loaders.append(data_loader)
            num_batches_list.append(len(data_loader))

        print ("createsamplers anddata loaders")

        # Create a list of data loader indices and shuffle them
        data_loader_indices = []
        for i, num_batches in enumerate(num_batches_list):
            data_loader_indices.extend([i] * num_batches)
        np.random.shuffle(data_loader_indices)

        data_loader_iters = [iter(loader) for loader in data_loaders]
        print ("create list of data loader indices")

        print (sum(num_batches_list))
        for overall_idx in tqdm(range(sum(num_batches_list))):
            data_loader_idx = data_loader_indices[overall_idx]
            data_loader_iter = data_loader_iters[data_loader_idx]
            try:
                (batch_data, batch_labels, batch_elevs, batch_azims) = next(data_loader_iter)
                #print ("not skip")
            except StopIteration:
                continue  # Skip if the iterator is exhausted

            feature_field_batch = features[data_loader_idx].unsqueeze(0).expand(batch_labels.shape[0], -1, -1)
            mesh = args.meshes[data_loader_idx]
            batch_indices[data_loader_idx] += 1

            # Apply attention and forward pass through the MLP
            weighted_vals = attention(feature_field_batch, batch_labels)
            input_tensor = torch.cat((feature_field_batch, weighted_vals), dim=-1)
            prob_tensor = mlp(input_tensor)
            #print ("apply attention")

            # Set probability per face and render
            setcolor_mesh_batched(mesh, prob_tensor)
            mesh.face_attributes = mesh.face_attributes.float()
            rendered_prob_views, elev, azim, mask = render_high.render_views_batched(
                mesh, show=False, std=args.frontview_std, return_views=True,
                center_azim=batch_azims, center_elev=batch_elevs, return_features=True, lighting=False,
                background=None, return_mask=True
            )
            #print ("set probability")
            # Calculate and backpropagate loss
            loss = masked_bce_loss(rendered_prob_views, batch_data, mask)
            loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                losses.append(loss.item())
            torch.cuda.empty_cache()
            print ("calculate loss")
            # Save checkpoint
            if sum(batch_indices) % args.save_interval == 0 or overall_idx == (sum(num_batches_list) - 1):
                torch.save({
                    'attention_state_dict': attention.state_dict(),
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'batch_indices': batch_indices,
                    'shuffled_indices_list': all_indices_list,
                    'losses': losses,
                    'num_batches': sum(num_batches_list)
                }, save_path)
                print(f'Checkpoint saved for epoch {epoch + 1}, batch indices {batch_indices}')
        #print ("iterate through index")

        if sum(num_batches_list) > 0:
            save_dir, fname = os.path.split(save_path)
            fbody, fext = fname.split(".")
            fname_epoch = ".".join(["%s_e%d" % (fbody, epoch + 1), fext])
            save_path_epoch = os.path.join(save_dir, fname_epoch)
            torch.save({
                'attention_state_dict': attention.state_dict(),
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'batch_indices': batch_indices,
                'shuffled_indices_list': all_indices_list,
                'losses': losses,
                'num_batches': sum(num_batches_list)
            }, save_path_epoch)
            print('Checkpoint saved for epoch number %d' % (epoch + 1))

        # Reset start batch indices for the next epoch
        start_batch_indices = [0] * len(datasets)


def save_or_append(filename, data):
    if os.path.exists(filename):
        existing_data = torch.load(filename, map_location=device)
        if len(data.shape) < len(existing_data.shape):
            combined_data = torch.cat([existing_data, data.unsqueeze(0)])
        else:
            combined_data = torch.cat([existing_data, data])
        torch.save(combined_data, filename)
    else:
        torch.save(data, filename)


def save_loss(loss, dir, name=None):
    plt.figure()
    plt.plot(loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Save the figure
    if name is not None:
        plt.title(name + ' over time')
        plt.savefig(os.path.join(dir, name + '.jpg'))
        plt.close()
    else:
        plt.title('Loss over time')
        plt.savefig(os.path.join(dir, 'loss.jpg'))
        plt.close()


def load_features(args):
    # Load features for multiple objects
    if args.mode == 'train':
        pred_f_list = []
        for path in args.encoder_f_paths:
            pred_f = torch.load(path)
            pred_f_list.append(pred_f)
        return pred_f_list
    elif args.mode == 'test':
        pred_f = torch.load(args.encoder_f_path_test)
        return pred_f


def masked_bce_loss(y_pred, y_true, mask):
    # Compute the raw BCE loss term-wise
    bce = F.binary_cross_entropy(y_pred, F.one_hot(y_true.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2).float(),
                                 reduction='none')
    mask = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), dim=1)
    # Apply the mask
    masked_bce = bce * mask

    # Compute the mean of the masked BCE values
    loss = masked_bce.sum() / mask.sum()
    return loss


def test_decoder(args, test_index):
    print('test decoder')
    num_gpus = torch.cuda.device_count()
    print('number of available gpus: %d' % num_gpus)

    render_high = Renderer(dim=(args.render_res, args.render_res))

    # MLP Settings: predicting 3D probability mask
    attention = ClickAttention(256, args).to(device)
    mlp = Decoder(depth=args.depth, width=[512] + [256] * args.depth, out_dim=2, input_dim=512,
                  positional_encoding=args.positional_encoding,
                  sigma=args.sigma).to(device)

    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        attention = torch.nn.DataParallel(attention, device_ids=device_ids_list)
        mlp = torch.nn.DataParallel(mlp, device_ids=device_ids_list)

    save_path = os.path.join(args.save_dir, args.model_name)
    print('decoder checkpoint path: %s' % save_path)

    checkpoint = torch.load(save_path, map_location=device)
    load_state_dict(attention, checkpoint['attention_state_dict'])
    load_state_dict(mlp, checkpoint['model_state_dict'])

    # Read learned 3D features
    pred_f = load_features(args)

    if len(test_index.shape) == 1:
        test_index = test_index.unsqueeze(0)

    batch_size = test_index.shape[0]

    feature_field_batch = pred_f.unsqueeze(0).expand(batch_size, -1, -1)
    start = time.time()

    # Click attention
    weighted_vals = attention(feature_field_batch, test_index)

    input_tensor = torch.cat((feature_field_batch, weighted_vals), dim=-1)

    # MLP
    prob_tensor = mlp(input_tensor)
    end = time.time()
    print('inference time: %.2f seconds' % (end - start))

    # Save vertex probability
    prob_tensor_np = prob_tensor.detach().cpu().numpy().astype(np.float32)
    prob_to_save = prob_tensor_np[0, :, 1]

    select_vertices_str = '_'.join(['v%d' % idx for idx in list(test_index[0])])
    save_name = 'vertex_probability_' + select_vertices_str + '.npy'
    save_path = os.path.join(args.save_dir, save_name)
    np.save(save_path, prob_to_save)
    print("vertex probability saved to %s" % save_path)

    # Save colored mesh
    test_index_list = list(test_index[0].cpu())
    prepare_colored_mesh(args.mesh_test, np.expand_dims(prob_to_save, 0), test_index_list, args)

    # Save render views
    test_azim_deg = [0, 60, 120, 180, 240, 300]
    test_elev_deg = [-60, -30, 0, 30, 60]
    render_views(args.mesh_test, test_index, prob_tensor, render_high, test_azim_deg, test_elev_deg, args)


def prepare_colored_mesh(mesh, prob_tensor, test_index, args):
    base_color = np.array([args.base_color], dtype=np.float32) / 255.
    seg_color = np.array([args.seg_color], dtype=np.float32) / 255.

    pos_color = np.array(args.pos_color, dtype=np.float32) / 255.
    neg_color = np.array(args.neg_color, dtype=np.float32) / 255.

    if args.show_seg:
        vertex_colors = prob_tensor.T * seg_color + (1. - prob_tensor.T) * base_color
    else:
        vertex_colors = np.ones_like(prob_tensor.T) * base_color

    mesh_o3d = mesh.export_to_open3d()
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    if not args.show_spheres:
        mesh_to_save = mesh_o3d
    else:
        sel_verts_idx = np.array(test_index, dtype=np.int32)
        sel_verts = mesh.vertices[np.abs(sel_verts_idx)].detach().cpu().numpy()

        col_verts = np.zeros_like(sel_verts)
        for i, v in enumerate(list(sel_verts_idx)):
            if v >= 0:
                col_verts[i] = pos_color
            else:
                col_verts[i] = neg_color

        mesh_to_save = add_spheres(copy.deepcopy(mesh_o3d), sel_verts, col_verts, save_mesh=False, save_path=None,
                                   sphere_radius=args.sphere_radius, sphere_resolution=10)

    verts = np.asarray(mesh_to_save.vertices)
    faces = np.asarray(mesh_to_save.triangles)
    colors = np.array(np.asarray(mesh_to_save.vertex_colors) * 255 + 0.5, dtype=np.int32)

    select_vertices_str = '_'.join(['v%d' % idx for idx in test_index])
    if not args.show_spheres:
        save_name = 'colored_mesh_' + select_vertices_str + '.ply'
    else:
        save_name = 'colored_mesh_with_clicks_' + select_vertices_str + '.ply'
    save_path = os.path.join(args.save_dir, save_name)
    write_ply(save_path, verts, faces, colors)
    print("mesh saved to %s" % save_path)

    return save_path


def render_views(mesh, test_index, prob_tensor, renderer, test_azim_deg, test_elev_deg, args):
    test_azim = np.array(test_azim_deg, dtype=np.float32) / 360 * 2 * np.pi
    test_elev = np.array(test_elev_deg, dtype=np.float32) / 180 * np.pi

    target_rgb = torch.zeros_like(mesh.vertices)  # initialize colors with zeros

    target_rgb[:] = torch.tensor([2. / 3., 2. / 3., 2. / 3.]).to(device)
    target_rgb[test_index[0, 0]] = torch.tensor([0., 255., 0.]).to(device)

    if test_index.shape[-1] > 1:
        if test_index[0, 1] >= 0:
            target_rgb[test_index[0, 1]] = torch.tensor([0., 255., 0.]).to(device)
        else:
            target_rgb[abs(test_index[0, 1])] = torch.tensor([255., 0., 0.]).to(device)

    azim_num = len(test_azim)
    elev_num = len(test_elev)
    _, axs = plt.subplots(elev_num, azim_num, figsize=(azim_num * 10, elev_num * 10))

    target_mesh = mesh
    for i in range(elev_num):
        for j in range(azim_num):
            setcolor_mesh_batched(target_mesh, target_rgb.unsqueeze(0))
            image_temp, elev, azim = renderer.render_views(mesh, num_views=1,
                                                           show=args.show,
                                                           std=args.frontview_std,
                                                           random_views=False,
                                                           center_elev=torch.tensor(test_elev[i:i + 1]).to(device),
                                                           center_azim=torch.tensor(test_azim[j:j + 1]).to(device),
                                                           lighting=True,
                                                           return_views=True,
                                                           background=torch.ones(3).to(device),
                                                           return_mask=False)
            image_temp1 = image_temp.squeeze(0).permute(1, 2, 0)
            selected_vertex_r, selected_vertex_g = find_red_green_pixels(image_temp1)

            setcolor_mesh_batched(mesh, prob_tensor)
            rendered_prob_views = renderer.render_views(mesh, num_views=1,
                                                        show=args.show,
                                                        std=args.frontview_std,
                                                        random_views=False,
                                                        center_elev=elev,
                                                        center_azim=azim,
                                                        lighting=False,
                                                        background=None,
                                                        return_mask=True)

            save_image_name = 'temp_view.png'
            save_image_path = os.path.join(args.save_dir, save_image_name)
            save_renders(args.save_dir, 0, image_temp, name=save_image_name)
            image = cv2.imread(save_image_path)
            axs[i, j].imshow(image)

            axs[i, j].set_xticks([])
            axs[i, j].set_xticks([], minor=True)
            axs[i, j].set_yticks([])
            axs[i, j].set_yticks([], minor=True)

            os.remove(save_image_path)

            show_mask(rendered_prob_views[0][0][1].detach().cpu().numpy(), axs[i, j])
            if len(selected_vertex_r) != 0:
                show_points(
                    np.array([[selected_vertex_r[0][1].detach().cpu().numpy(), selected_vertex_r[0][0].detach().cpu().numpy()]]),
                    np.array([0]), axs[i, j], marker_size=307)
            if len(selected_vertex_g) != 0:
                show_points(
                    np.array([[selected_vertex_g[0][1].detach().cpu().numpy(), selected_vertex_g[0][0].detach().cpu().numpy()]]),
                    np.array([1]), axs[i, j], marker_size=307)

    select_vertices_str = '_'.join(['v%d' % idx for idx in list(test_index[0])])
    save_name = 'render_views_' + select_vertices_str + '.png'
    save_path = os.path.join(args.save_dir, save_name)

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()
    print("render views saved to %s" % save_path)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)

    # Directory structure
    parser.add_argument('--obj_paths', nargs='+', type=str, default=['./meshes/hammer.obj'])
    parser.add_argument('--obj_path_test', type=str, default='./meshes/hammer.obj')
    parser.add_argument('--encoder_f_paths', nargs='+', type=str,
                        default=['./experiments/hammer/encoder/pred_f.pth'])
    parser.add_argument('--encoder_f_path_test', type=str, default='./experiments/hammer/encoder/pred_f.pth')
    parser.add_argument('--decoder_data_dirs', nargs='+', type=str, default=['./data/hammer/decoder_data'])
    parser.add_argument('--save_dir', type=str, default='./experiments/hammer/decoder/')
    parser.add_argument('--model_name', type=str, default='decoder_checkpoint.pth')

    # Training data setting
    parser.add_argument('--use_positive_click', type=int, default=0)
    parser.add_argument('--use_negative_click', type=int, default=0)

    # Mesh + data info
    parser.add_argument('--names', nargs='+', type=str, default=['hammer'])
    parser.add_argument('--name_test', type=str, default='hammer')
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--views_per_vert', type=int, default=100)

    # Render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[3.14, 0.])
    parser.add_argument('--render_res', type=int, default=224)

    # Attention
    parser.add_argument('--use_attention_q', type=int, default=1)
    parser.add_argument('--use_attention_k', type=int, default=1)
    parser.add_argument('--use_attention_v', type=int, default=1)
    parser.add_argument('--redsidual_attention', type=int, default=0)
    parser.add_argument('--scale_attention', type=int, default=1)

    # Network
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256)  # 256 channels for SAM embedding feature
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)

    # Optimization
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--return_original', type=int, default=0)

    # Parallel, multi-GPU training
    parser.add_argument('--use_data_parallel', type=int, default=0)

    # Mode
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--select_vertices', nargs='+', type=int, default=[0])
    parser.add_argument('--show', type=int, default=0)

    # Visualization
    parser.add_argument('--base_color', nargs=3, type=int, default=[180, 180, 180])
    parser.add_argument('--show_seg', type=int, default=1)
    parser.add_argument('--seg_color', nargs=3, type=int, default=[60, 160, 250])

    parser.add_argument('--show_spheres', type=int, default=0)
    parser.add_argument('--sphere_radius', type=float, default=0.025)
    parser.add_argument('--pos_color', nargs=3, type=int, default=[0, 255, 0])
    parser.add_argument('--neg_color', nargs=3, type=int, default=[255, 0, 0])

    args = parser.parse_args()

    # Create decoder directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "train":
        # Load meshes for multiple objects
        args.meshes = []
        for i, obj_path in enumerate(args.obj_paths):
            mesh = loadmesh(dir=obj_path, name=args.names[i], load_rings=True)
            args.meshes.append(mesh)
        train_decoder(args)

    if args.mode == "test":
        args.mesh_test = loadmesh(dir=args.obj_path_test, name=args.name_test, load_rings=True)
        test_decoder(args, test_index=torch.tensor(args.select_vertices))
