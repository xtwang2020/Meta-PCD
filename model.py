
import torch
import torch.nn as nn
import torch.nn.functional as F



class PENet(nn.Module):
    def __init__(self, dim, num_points):
        super(PENet, self).__init__()
        self.num_points = num_points
        self.dim = dim
        self.displace_conv1 = nn.Conv1d(20, 128, 1)
        self.displace_conv2 = nn.Conv1d(128, 256, 1)
        self.displace_conv3 = nn.Conv1d(256, 512, 1)
        self.displace_bn1 = nn.BatchNorm1d(128)
        self.displace_bn2 = nn.BatchNorm1d(256)
        self.displace_bn3 = nn.BatchNorm1d(512)
        self.displace_P1 = nn.MaxPool1d(500)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.displace_conv4_1 = nn.Conv1d(1536, 512, 1)
        self.displace_conv4_2 = nn.Conv1d(1536, 512, 1)
        self.displace_conv4_3 = nn.Conv1d(1536, 512, 1)
        self.displace_conv4_4 = nn.Conv1d(1536, 512, 1)

        self.displace_conv5_1 = nn.Conv1d(512, 64, 1)
        self.displace_conv5_2 = nn.Conv1d(512, 64, 1)
        self.displace_conv5_3 = nn.Conv1d(512, 64, 1)
        self.displace_conv5_4 = nn.Conv1d(512, 64, 1)

        self.displace_conv6_1 = nn.Conv1d(64, self.dim, 1)
        self.displace_conv6_2 = nn.Conv1d(64, self.dim, 1)
        self.displace_conv6_3 = nn.Conv1d(64, self.dim, 1)
        self.displace_conv6_4 = nn.Conv1d(64, self.dim, 1)

        self.displace_bn4_1 = nn.BatchNorm1d(512)
        self.displace_bn4_2 = nn.BatchNorm1d(512)
        self.displace_bn4_3 = nn.BatchNorm1d(512)
        self.displace_bn4_4 = nn.BatchNorm1d(512)

        self.displace_bn5_1 = nn.BatchNorm1d(64)
        self.displace_bn5_2 = nn.BatchNorm1d(64)
        self.displace_bn5_3 = nn.BatchNorm1d(64)
        self.displace_bn5_4 = nn.BatchNorm1d(64)

        self.displace_bn6_1 = nn.BatchNorm1d(self.dim, )
        self.displace_bn6_2 = nn.BatchNorm1d(self.dim, )
        self.displace_bn6_3 = nn.BatchNorm1d(self.dim, )
        self.displace_bn6_4 = nn.BatchNorm1d(self.dim, )

        self.displace_conv4 = [self.displace_conv4_1,
                               self.displace_conv4_2,
                               self.displace_conv4_3,
                               self.displace_conv4_4,
                               ]

        self.displace_conv5 = [self.displace_conv5_1,
                               self.displace_conv5_2,
                               self.displace_conv5_3,
                               self.displace_conv5_4,

                               ]
        self.displace_conv6 = [self.displace_conv6_1,
                               self.displace_conv6_2,
                               self.displace_conv6_3,
                               self.displace_conv6_4,

                               ]
        self.displace_bn4 = [self.displace_bn4_1,
                             self.displace_bn4_2,
                             self.displace_bn4_3,
                             self.displace_bn4_4,

                             ]
        self.displace_bn5 = [self.displace_bn5_1,
                             self.displace_bn5_2,
                             self.displace_bn5_3,
                             self.displace_bn5_4,

                             ]
        self.displace_bn6 = [self.displace_bn6_1,
                             self.displace_bn6_2,
                             self.displace_bn6_3,
                             self.displace_bn6_4,
                             ]
        self.meta_conv1 = nn.Conv1d(20, 32, 1)
        self.meta_conv2 = nn.Conv1d(32, 128, 1)
        self.meta_conv3 = nn.Conv1d(128, 256, 1)
        self.meta_bn1 = nn.BatchNorm1d(32)
        self.meta_bn2 = nn.BatchNorm1d(128)
        self.meta_bn3 = nn.BatchNorm1d(256)
    def attention_net(self, lstm_output, final_state):
        n_hidden = 512
        hidden = final_state.view(-1, n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        return soft_attn_weights  # soft_attn_weights : [batch_size, n_hidden]

    def forward(self, input):
        # input:[x, y, z]
        input = input.transpose(1, 2)
        Meta_0 = torch.ones(input.shape).cuda()[:,:1,:]

        Meta_x = input[:,:1,:]
        Meta_y = input[:,1:2,:]
        Meta_z = input[:,2:3,:]

        Meta_2 = input*input
        Meta_xx = Meta_2[:,:1,:]
        Meta_yy = Meta_2[:,1:2,:]
        Meta_zz = Meta_2[:,2:3,:]
        Meta_xy = Meta_x*Meta_y
        Meta_yz = Meta_y*Meta_z
        Meta_xz = Meta_x*Meta_z

        Meta_3 = input*input*input
        Meta_xxx = Meta_3[:,:1,:]
        Meta_xxy = Meta_xx*Meta_y
        Meta_xxz = Meta_xx*Meta_z
        Meta_xyy = Meta_x*Meta_yy
        Meta_xyz = Meta_x*Meta_y*Meta_z
        Meta_xzz = Meta_x*Meta_zz
        Meta_yyy = Meta_3[:,1:2,:]
        Meta_yyz = Meta_yy*Meta_z
        Meta_yzz = Meta_y*Meta_zz
        Meta_zzz = Meta_3[:,2:3,:]

        Meta_input=torch.cat([Meta_0,Meta_x,Meta_y,Meta_z,Meta_xx,Meta_xy,Meta_xz,Meta_yy,Meta_yz,Meta_zz,
                        Meta_xxx,Meta_xxy,Meta_xxz,Meta_xyy,Meta_xyz,Meta_xzz,Meta_yyy,Meta_yyz,Meta_yzz,Meta_zzz],1)

        x_encode = F.relu6(self.displace_bn1(self.displace_conv1(Meta_input)))
        x_encode = F.relu6(self.displace_bn2(self.displace_conv2(x_encode)))
        x_encode = F.relu6(self.displace_bn3(self.displace_conv3(x_encode)))

        center_feature = torch.unsqueeze(x_encode[..., 0], -1)
        x_encode = x_encode - center_feature
        max_feature = self.displace_P1(x_encode)
        x_encode = torch.cat([x_encode,
                              torch.Tensor.repeat(center_feature, [1, 1, x_encode.size()[-1]]),
                              torch.Tensor.repeat(max_feature, [1, 1, x_encode.size()[-1]])], dim=-2)



        Meta = F.relu6(self.meta_bn1(self.meta_conv1(Meta_input)))
        Meta = F.relu6(self.meta_bn2(self.meta_conv2(Meta)))
        Meta = F.relu6(self.meta_bn3(self.meta_conv3(Meta)))
        displace = []
        for i, meta in enumerate([Meta, Meta, Meta]):
            x = F.relu6(self.displace_bn4[i](self.displace_conv4[i](x_encode)))
            coeficients = F.relu6(self.displace_bn5[i](self.displace_conv5[i](x)))
            coeficients = self.displace_bn6[i](self.displace_conv6[i](coeficients))
            coeficients = F.dropout(coeficients, p=0.2)
            xyz_i = torch.mean(meta * coeficients, -1, keepdim=True)
            xyz_i = torch.mean(xyz_i, 1, keepdim=True)
            displace.append(xyz_i)
        displace = torch.cat(displace, 1)
        return displace

