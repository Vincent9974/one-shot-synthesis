# import torch
# import numpy as np
# from core.differentiable_augmentation.torch_utils import misc
#
# def matrix(*rows, device=None):
#     assert all(len(row) == len(rows[0]) for row in rows)
#     elems = [x for row in rows for x in row]
#     ref = [x for x in elems if isinstance(x, torch.Tensor)]
#     if len(ref) == 0:
#         return misc.constant(np.asarray(rows), device=device)
#     assert device is None or device == ref[0].device
#     elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
#     return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))
#
# cx = (80 - 1) / 2
# cy = (80 - 1) / 2
# cz = (80 - 1) / 2
# cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1])
# cp2 = matrix([-cx, -cy, -cz, 1],[-cx,  cy,  cz, 1],[-cx,  cy, -cz, 1],
#             [-cx, -cy,  cz, 1],[ cx, -cy, -cz, 1],[ cx,  cy, -cz, 1],
#             [-cx, -cy,  cz, 1],[ cx,  cy,  cz, 1])
# # print(cp)
# # tensor([[-39.5000, -39.5000,   1.0000], [4,3]
# # [ 39.5000, -39.5000,   1.0000],
# # [ 39.5000,  39.5000,   1.0000],
# # [-39.5000,  39.5000,   1.0000]])
# margin = cp[:, :2].permute(1, 0).flatten(1)
# margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
# # margin2 = cp2[:, :3].permute(1, 0).flatten(1)
# # margin = torch.cat([-margin, margin]).max(dim=1).values
# print(margin)




# print(cp) [8,4]
# tensor([[-39.5000, -39.5000, -39.5000,   1.0000],
#         [-39.5000,  39.5000,  39.5000,   1.0000],
#         [-39.5000,  39.5000, -39.5000,   1.0000],
#         [-39.5000, -39.5000,  39.5000,   1.0000],
#         [ 39.5000, -39.5000, -39.5000,   1.0000],
#         [ 39.5000,  39.5000, -39.5000,   1.0000],
#         [-39.5000, -39.5000,  39.5000,   1.0000],
#         [ 39.5000,  39.5000,  39.5000,   1.0000]])


from PIL import Image, TiffImagePlugin
def test_custom_metadata():

    img = Image.open('/home1/Usr/zhangwenqing/SIV-GAN-3D/datasets/berea_3d/image/berea_16.tif')

    info = TiffImagePlugin.ImageFileDirectory()
    CustomTagId = 37000

    info[CustomTagId] = 6
    info.tagtype[CustomTagId] = 3 # 'short' TYPE

    Image.DEBUG=True
    TiffImagePlugin.WRITE_LIBTIFF = False # Set to True to see it break.
    img.save('/home1/Usr/zhangwenqing/SIV-GAN-3D/datasets/berea_3d/image/berea_16meta.tif', tiffinfo = info)

test_custom_metadata()