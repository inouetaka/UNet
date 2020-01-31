# U_Net Pytorch

## dataset
data/   
 ├ imgs - *.jpg   
 └ masks - *.gif   

## Train
オプション
```
'-e', '--epochs', metavar='E', type=int, default=5 
'-b', '--batch-size', metavar='B', type=int, nargs='?', default=1
'-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1
'-f', '--load', dest='load', type=str, default=False
'-s', '--scale', dest='scale', type=float, default=0.5
'-v', '--validation', dest='val', type=float, default=10.0
```

## Predict
オプション
```
'--model', '-m', default='MODEL.pth'
'--input', '-i', metavar='INPUT', nargs='+', required=True
'--output', '-o', metavar='INPUT', nargs='+'
'--viz', '-v', action='store_true', default=False
'--no-save', '-n', action='store_true', default=False
'--mask-threshold', '-t', type=float, default=0.5
'--scale', '-s', type=float, default=0.5
```
