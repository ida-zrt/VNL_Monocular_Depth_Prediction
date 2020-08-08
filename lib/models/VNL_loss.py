import torch
import torch.nn as nn
import numpy as np


class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self,
                 focal_x,
                 focal_y,
                 input_size,
                 optical_center,
                 delta_cos=0.867,
                 delta_diff_x=0.01,
                 delta_diff_y=0.01,
                 delta_diff_z=0.01,
                 z_thres=0.0001,
                 sample_ratio=0.15,
                 only_valid=True):
        super(VNL_Loss, self).__init__()
        # camera params to reconstruct 3D with depth
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(optical_center[0], dtype=torch.float32).cuda()
        self.v0 = torch.tensor(optical_center[1], dtype=torch.float32).cuda()
        self.init_image_coor()

        # thresholds for filtering out 3D points
        self.delta_cos = delta_cos  # linear threshold, filter out 3 co-linear points in 3D
        self.delta_diff_x = delta_diff_x  # x, y, z thresholds for filtering out points that are too close
        self.delta_diff_y = delta_diff_y  # notice: the depth is often scaled, so it should also be scaled
        self.delta_diff_z = delta_diff_z  # after being determined by real distance in reality
        self.z_thres = z_thres  # threshold for filtering out invalid depths

        self.sample_ratio = sample_ratio  # sampling ratio, increase it if you have too many invalid depth
        self.only_valid = only_valid

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0,
                          self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int)
        p123 = {
            'p1_x': p1_x,
            'p1_y': p1_y,
            'p2_x': p2_x,
            'p2_y': p2_y,
            'p3_x': p3_x,
            'p3_y': p3_y
        }
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([
            pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :,
                                                                    np.newaxis]
        ], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz):
        pw = self.form_pw_groups(p123, gt_xyz)
        # pt1 = np.array([[0, 0, 0]], dtype=float)
        # pt2 = np.array([[1, 1, 1]], dtype=float)
        # pt3 = np.array([[2, 2, 50]], dtype=float)
        # pw = np.concatenate([pt1, pt2, pt3], axis=0)
        # pw = np.transpose(pw.reshape((1, 1, 3, 3)), (0, 1, 3, 2))
        # pw = torch.tensor(pw).cuda()
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]

        # ignore linear
        pw_diff = torch.cat([
            pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis],
            pw23[:, :, :, np.newaxis]
        ], 3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(
            0, 2, 1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1,
                                index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        # calculate norm product, organized in matrix
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1),
                       q_norm.view(m_batchsize * groups, 1, index))
        # calculate dot product, organized in matrix
        energy = torch.bmm(proj_query,
                           proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        # calculate cos values between eacc vector(pw_diff)
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)

        # create mask by cos values, if not linear this sum should be 3
        mask_cos = torch.sum((norm_energy > self.delta_cos) +
                             (norm_energy < -self.delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)

        # ignore padding and invilid depth
        # if this sum is 3, it means all points are valid in this group
        if not self.only_valid:
            mask_pad = torch.sum(pw[:, :, 2, :] > self.z_thres, 2) == 3

        # ignore near
        mask_x = torch.sum(
            torch.abs(pw_diff[:, :, 0, :]) < self.delta_diff_x, 2) > 0
        mask_y = torch.sum(
            torch.abs(pw_diff[:, :, 1, :]) < self.delta_diff_y, 2) > 0
        mask_z = torch.sum(
            torch.abs(pw_diff[:, :, 2, :]) < self.delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore

        if not self.only_valid:
            mask = mask_pad & mask_near
        else:
            mask = mask_near

        return mask, pw

    def select_valid_index(self, gt_depth):
        validIndex = (gt_depth > self.z_thres).nonzero().cpu().numpy()

        p1_selector = np.random.choice(
            np.arange(validIndex.shape[0]),
            int(validIndex.shape[0] * self.sample_ratio))
        p1_x = validIndex[p1_selector, 3]
        p1_y = validIndex[p1_selector, 2]

        p2_selector = np.random.choice(
            np.arange(validIndex.shape[0]),
            int(validIndex.shape[0] * self.sample_ratio))
        p2_x = validIndex[p2_selector, 3]
        p2_y = validIndex[p2_selector, 2]

        p3_selector = np.random.choice(
            np.arange(validIndex.shape[0]),
            int(validIndex.shape[0] * self.sample_ratio))
        p3_x = validIndex[p3_selector, 3]
        p3_y = validIndex[p3_selector, 2]

        p123 = {
            'p1_x': p1_x.astype(np.int),
            'p1_y': p1_y.astype(np.int),
            'p2_x': p2_x.astype(np.int),
            'p2_y': p2_y.astype(np.int),
            'p3_x': p3_x.astype(np.int),
            'p3_y': p3_y.astype(np.int)
        }
        return p123

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        if self.only_valid:
            p123 = self.select_valid_index(gt_depth)
        else:
            p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3,
                                                   -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(
            1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(
            1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    import cv2
    vnl_loss = VNL_Loss(1.0, 1.0, (480, 640))
    pred_depth = np.ones([2, 1, 480, 640])
    gt_depth = np.ones([2, 1, 480, 640])
    gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
    loss = vnl_loss(pred_depth, gt_depth)
    print(loss)
