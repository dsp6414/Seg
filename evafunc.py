import numpy as np
import os
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt


def main():
    algs = ['duts_res50']
    datasets = ['ECSSD']
    for dataset in datasets:
        print(dataset)
        dir = '/home/zeng/data/datasets/saliency_Dataset/%s'%dataset
        output_dir = './%s'%dataset
        gt_dir = '%s/masks'%dir
        input_dirs = ['%s/%s'%(dir, alg) for alg in algs]
        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for input_dir, alg in zip(input_dirs, algs):
            evaluate(input_dir, gt_dir, output_dir, alg)
            sb = np.load('%s/%s.npz'%(output_dir, alg))
            ax1.plot(sb['m_recs'], sb['m_pres'], linewidth=1, label=alg)
            ax2.plot(np.linspace(0, 1, 21), sb['m_fms'], linewidth=1, label=alg)
            print('%s, fm: %.4f, mea: %.4f'%(alg, sb['m_thfm'], sb['m_mea']))
        ax1.grid(True)
        ax1.set_xlabel('Recall', fontsize=14)
        ax1.set_ylabel('Precision', fontsize=14)
        ax2.grid(True)
        ax2.set_xlabel('Threshold', fontsize=14)
        ax2.set_ylabel('F-measure', fontsize=14)
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(0.5, -0.5), ncol=8, fontsize=14)
        fig.savefig('%s.pdf'%dataset, bbox_extra_artists=(lgd,), bbox_inches='tight')


def evaluate(input_dir, gt_dir, output_dir=None, name=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filelist = os.listdir(input_dir)

    eps = np.finfo(float).eps

    m_pres = np.zeros(21)
    m_recs = np.zeros(21)
    m_fms = np.zeros(21)
    m_thfm = 0
    m_mea = 0
    it = 1
    for filename in filelist:
        if not filename.endswith('.png'):
            continue
        # print('evaluating image %d'%it)
        mask = Image.open('%s/%s' % (input_dir, filename))
        gt = Image.open('%s/%s' % (gt_dir, filename))
        mask = mask.resize(gt.size)
        mask = np.array(mask, dtype=np.float)
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        mask = (mask - mask.min()) / (mask.max()-mask.min()+eps)
        gt = np.array(gt, dtype=np.uint8)
        gt[gt != 0] = 1
        pres = []
        recs = []
        fms = []
        mea = np.abs(gt-mask).mean()
        # threshold fm
        binary = np.zeros(mask.shape)
        th = 2*mask.mean()
        if th > 1:
            th = 1
        binary[mask >= th] = 1
        sb = (binary * gt).sum()
        pre = sb / (binary.sum()+eps)
        rec = sb / (gt.sum()+eps)
        thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
        for th in np.linspace(0, 1, 21):
            binary = np.zeros(mask.shape)
            binary[ mask >= th] = 1
            pre = (binary * gt).sum() / (binary.sum()+eps)
            rec = (binary * gt).sum() / (gt.sum()+ eps)
            fm = 1.3 * pre * rec / (0.3*pre + rec + eps)
            pres.append(pre)
            recs.append(rec)
            fms.append(fm)
        fms = np.array(fms)
        pres = np.array(pres)
        recs = np.array(recs)
        m_mea = m_mea * (it-1) / it + mea / it
        m_fms = m_fms * (it - 1) / it + fms / it
        m_recs = m_recs * (it - 1) / it + recs / it
        m_pres = m_pres * (it - 1) / it + pres / it
        m_thfm = m_thfm * (it - 1) / it + thfm / it
        it += 1
    if not (output_dir is None or name is None):
        np.savez('%s/%s.npz'%(output_dir, name), m_mea=m_mea, m_thfm=m_thfm, m_recs=m_recs, m_pres=m_pres, m_fms=m_fms)
    return m_thfm, m_mea


if __name__ == '__main__':
    main()



