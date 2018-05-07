# 二值分割
趁电脑空闲，试一下vgg, resnet系列和densenet系列的效果

## 想到了一个办法

```python
        try:
            for ib, (data, lbl) in enumerate(train_loader):
                # 一个很久的epoch
        except KeyboardInterrupt:
            filename = ('%s/model_int.pth' % (check_dir))
            torch.save(model.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))
            continue
```

## Adam 

duts_vgg, fm: 0.8270, mea: 0.0754

duts_res101, fm: 0.7080, mea: 0.1451 （感觉这个有毛病，换下优化参数试试）

duts_dense121, fm: 0.8513, mea: 0.0607

duts_dense161, fm: 0.8537, mea: 0.0584

duts_dense169, fm: 0.8436, mea: 0.0601, loss 0.03~0.04

duts_dense201, fm: 0.8680, mea: 0.0518, loss 0.03

## SGD

duts_res50, fm: 0.8213, mea: 0.0699

resnet101 fm: 0.8477, mea: 0.0599, loss 0.05~0.06

duts_res152, fm: 0.8566, mea: 0.0565

densenet121 duts_dense121, fm: 0.8476, mea: 0.0637

vgg duts_sgd_vgg, fm: 0.7813, mea: 0.0924, loss 0.08~0.09