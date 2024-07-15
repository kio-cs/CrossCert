import numpy
import torch
torch.set_printoptions(profile="full")


def predict_and_certify(inpt, net, block_size, size_to_certify, num_classes, threshold=0.0):
    predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    # smoothing happen in here
    for pos in range(batch.shape[2]):

        out_c1 = torch.zeros(batch.shape).cuda()
        out_c2 = torch.zeros(batch.shape).cuda()
        if (pos + block_size > batch.shape[2]):
            out_c1[:, :, pos:] = batch[:, :, pos:]
            out_c2[:, :, pos:] = 1. - batch[:, :, pos:]

            out_c1[:, :, :pos + block_size - batch.shape[2]] = batch[:, :, :pos + block_size - batch.shape[2]]
            out_c2[:, :, :pos + block_size - batch.shape[2]] = 1. - batch[:, :, :pos + block_size - batch.shape[2]]
        else:
            out_c1[:, :, pos:pos + block_size] = batch[:, :, pos:pos + block_size]
            out_c2[:, :, pos:pos + block_size] = 1. - batch[:, :, pos:pos + block_size]

        out_c1 = out_c1.permute(0, 3, 1, 2)
        out_c2 = out_c2.permute(0, 3, 1, 2)
        out = torch.cat((out_c1, out_c2), 1)
        softmx = torch.nn.functional.softmax(net(out), dim=1)
        thresh, predicted = torch.nn.functional.softmax(net(out), dim=1).max(1)
        # print(thresh)
        predictions += (softmx >= threshold).type(torch.int).cuda()
    predinctionsnp = predictions.cpu().numpy()
    idxsort = numpy.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -numpy.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    num_affected_classifications = (size_to_certify + block_size - 1)
    cert = torch.tensor(((val - valsecond > 2 * num_affected_classifications) | (
            (val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
    return torch.tensor(idx).cuda(), cert


def gen_ablation_set_row_fix(ablation_size,image_size=224):
    # generate a R-covering mask set
    # calculate mask size

    ablation_list = []
    idx_list1 = list(range(0,image_size))

    for x in idx_list1:
        mask = torch.zeros([1,1,image_size,image_size],dtype=bool).cuda()
        if x+ablation_size>image_size:
            mask[..., x:x + ablation_size, :] = True
            mask[..., 0:0 + ablation_size + x - image_size, :] = True
        else:
            mask[..., x:x + ablation_size, :] = True
        ablation_list.append(mask)

    return ablation_list, ablation_size,1


def gen_ablation_set_column_fix(ablation_size,image_size=224):
    # generate a R-covering mask set
    # calculate mask size

    ablation_list = []
    idx_list1 = list(range(0,image_size))

    for x in idx_list1:
        mask = torch.zeros([1,1,image_size,image_size],dtype=bool).cuda()
        if x+ablation_size>image_size:
            mask[..., :, x:x + ablation_size] = True
            mask[..., :, 0:0 + ablation_size + x - image_size] = True
        else:
            mask[..., :, x:x + ablation_size] = True
        ablation_list.append(mask)
    return ablation_list, ablation_size,1

def gen_ablation_set_xie(ablation_size,image_size=224):
    # generate a R-covering mask set
    # calculate mask size

    ablation_list = []
    idx_list1 = list(range(0,image_size))

    for x in idx_list1:
        mask = torch.zeros([1,1,image_size,image_size],dtype=bool).cuda()
        if x+ablation_size>image_size:
            mask[..., x:x + ablation_size+1, x:x + ablation_size] = True
            mask[..., :, 0:0 + ablation_size + x - image_size] = True
        else:
            mask[..., x:x + ablation_size+1, x:x + ablation_size] = True
        ablation_list.append(mask)
    return ablation_list, ablation_size,1


def gen_ablation_set_block(ablation_size,image_size=224):
    # generate a R-covering mask set
    # calculate mask size
    ablation_list = []
    idx_list1 = list(range(0,image_size))
    idx_list2 = list(range(0,image_size))

    for x in idx_list1:
        for y in idx_list2:
            mask = torch.zeros([1,1,image_size,image_size],dtype=bool).cuda()
            if x+ablation_size>image_size and y+ablation_size>image_size:
                mask[..., x:x + ablation_size, y:y + ablation_size] = True
                mask[..., 0:0 + ablation_size + x - image_size, 0:0 + ablation_size + y - image_size] = True
            elif y+ablation_size>image_size:
                mask[..., x:x + ablation_size, y:y + ablation_size] = True
                mask[..., x:x + ablation_size, 0:0 + ablation_size + y - image_size] = True
            elif x+ablation_size>image_size:
                mask[..., x:x + ablation_size, y:y + ablation_size] = True
                mask[..., 0:0 + ablation_size + x - image_size, y:y + ablation_size] = True
            else:
                mask[..., x:x + ablation_size, y:y + ablation_size] = True

            ablation_list.append(mask)
    return ablation_list, ablation_size,1


def random_one_ablation(ablation_list):
    idx=numpy.random.randint(0,len(ablation_list))
    return ablation_list[idx]