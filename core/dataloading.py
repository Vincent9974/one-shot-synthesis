import os
import tifffile
import torch
import warnings
# from PIL import Image
# from torchvision import transforms as TR
# import torchvision.transforms.functional as F
from .recommended_config import get_recommended_config
#from PIL import image

def prepare_dataloading(opt):
    dataset = Dataset(opt)
    recommended_config = {"image resolution": dataset.image_resolution,#（80，80，80）
                          "noise_shape": dataset.recommended_config[0], #（5，5，5）
                          "num_blocks_g":  dataset.recommended_config[1],#5
                          "num_blocks_d":  dataset.recommended_config[2],#6
                          "num_blocks_d0": dataset.recommended_config[3],#2
                          "no_masks": dataset.no_masks,
                          "num_mask_channels": dataset.num_mask_channels}#None
    if not recommended_config["no_masks"] and not opt.no_masks:
        print("Using the training regime *with* segmentation masks")
    else:
        opt.no_masks = True
        print("Using the training regime *without* segmentation masks")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True, num_workers=2)#PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    #dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=8)
    return dataloader, recommended_config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        """
        The dataset class. Supports both regimes *with* and *without* segmentation masks.
        """
        self.device = opt.device
        # --- images --- #
        self.root_images = os.path.join(opt.dataroot, opt.dataset_name, "image")
        self.root_masks = os.path.join(opt.dataroot, opt.dataset_name, "mask")#没有，所以没用
        #print(self.get_frames_list(self.root_images))
        self.list_imgs = self.get_frames_list(self.root_images)
        #print(self.list_imgs)
        assert len(self.list_imgs) > 0, "Found no images"
        self.image_resolution, self.recommended_config = get_recommended_config(self.get_im_resolution(opt.max_size))
        # print(self.image_resolution)
        # print(self.recommended_config)

        # --- masks --- #
        if os.path.isdir(self.root_masks) and not opt.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        else:
            self.no_masks = True
            self.num_mask_channels = None

        print("Created a dataset of size =", len(self.list_imgs), "with image resolution", self.image_resolution)

    def get_frames_list(self, path):
        return sorted(os.listdir(path))

    def __len__(self):
        return 100000000  # so first epoch finishes only with break

    def get_im_resolution(self, max_size):
        """
        Iterate over images to determine image resolution.
        If there are images with different shapes, return the square of average size
        """
        res_list = list()
        for cur_img in self.list_imgs:
            #img_pil = Image.open(os.path.join(self.root_images, cur_img)).convert("RGB")
            #img_pil = Image.open(os.path.join(self.root_images, cur_img))
            img_pil = tifffile.imread(os.path.join(self.root_images, cur_img))
            print(img_pil.shape)
            #res_list.append(img_pil.size)
            res_list.append(img_pil.shape)
            #print(res_list)
        all_res_equal = len(set(res_list)) <= 1  #set将重复值的单个副本存储到其中
        if all_res_equal:
            size_1, size_2, size_3 = res_list[0]  # all images have same resolution -> using original resolution
        else:
            warnings.warn("Images in the dataset have different resolutions. Resizing them to squares of mean size.")
            size_1 = size_2 = size_3 = sum([sum(item) for item in res_list]) / (2 * len(res_list))
        size_1, size_2, size_3 = self.bound_resolution(size_1, size_2, size_3,max_size)
        return size_3, size_2, size_1

    def bound_resolution(self, size_1, size_2, size_3, max_size):
        """
        Ensure the image shape does not exceed --max_size
        """
        if size_1 > max_size:
            size_1, size_2, size_3 = max_size, size_2 / (size_1 / max_size), size_3 / (size_1 / max_size)
        if size_2 > max_size:
            size_1, size_2, size_3 = size_1 / (size_2 / max_size), max_size, size_3 / (size_2 / max_size)
        if size_3 > max_size:
            size_1, size_2, size_3 = size_1 / (size_3 / max_size), size_2 / (size_3 / max_size), max_size
        return int(size_1), int(size_2), int(size_3)

    # def get_num_mask_channels(self): #没用
    #     """
    #     Iterate over all masks to determine how many classes are there
    #     """
    #     max_index = 0
    #     for cur_mask in self.list_masks:
    #         im = TR.functional.to_tensor(Image.open(os.path.join(self.root_masks, cur_mask)))
    #         if (im.unique() * 256).max() > 30:
    #             # --- black-white map of one object and background --- #
    #             max_index = 2 if max_index < 2 else max_index
    #         else:
    #             # --- multiple semantic objects --- #
    #             cur_max = torch.max(torch.round(im * 256))
    #             max_index = cur_max + 1 if max_index < cur_max + 1 else max_index
    #     return int(max_index)

    # def create_mask_channels(self, mask): #没用
    #     """
    #     Convert a mask to one-hot representation
    #     """
    #     if (mask.unique() * 256).max() > 30:
    #         # --- only object and background--- #
    #         mask = torch.cat((1 - mask, mask), dim=0)
    #         return mask
    #     else:
    #         # --- multiple semantic objects --- #
    #         integers = torch.round(mask * 256)
    #         mask = torch.nn.functional.one_hot(integers.long(), num_classes=self.num_mask_channels) #self.num_mask_channels=0
    #         mask = mask.float()[0].permute(2, 0, 1)
    #         return mask

    def __getitem__(self, index):
        output = dict()
        idx = index % len(self.list_imgs)
        #print(index)
        #print(idx)
        target_size = self.image_resolution
        #print(target_size)

        # --- image ---#
        #img_pil = Image.open(os.path.join(self.root_images, self.list_imgs[idx])).convert("RGB")


        img_pil = tifffile.imread(os.path.join(self.root_images, self.list_imgs[idx]))
        #img_pil=img_pil.transform.ToPILImage()
       # print(img_pil)
       # img = F.to_tensor(F.resize(img_pil, size=target_size))
        #print(img_pil)
        #print(img_pil.shape)

        # x = img_pil[:, :, :,:, None]  # ！!多：
        # print(x)
        # x = x.transpose((4, 3, 0, 1, 2)) / 255
        # x = torch.from_numpy(x)
        # x = x.to(torch.device('cuda'))
        # x=x.type(torch.cuda.FloatTensor)
        # x = (x - 0.5) * 2
        # x=x.clamp(-1, 1)
        # x = x[:, 0:3, :, :, :]

        # print(img_pil)
        # print(img_pil.shape)
        # print(img_pil.size)
        #img = F.to_tensor(F.resize(img_pil, size=target_size))
        #img = torch.from_numpy(torch.resize(img_pil, size=target_size))
        img = torch.from_numpy(img_pil)
        img = img.unsqueeze(0)  #增加一个维度torch.size([1,80,80,80])
        img = img.expand(3,16,16,16)  #对shape为1的进行扩展，对shape不为1的只能保持不变
        #print(img)
        #img = img.unsqueeze(0)
        #print(img.shape)
        #print(img)
        img = (img/255 - 0.5) * 2
        #print(img)
        output["images"] = img #output是个字典，key:"images",value:img(这里的img是张量)
        # print(output["images"])
        # print(output["images"].shape)

        # --- mask ---#
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        return output



