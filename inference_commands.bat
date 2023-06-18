@REM CAM minmax
python .\inference.py -m model\CAM_10e-5\epoch_22.pth -t CAM -s 98 -n minmax

@REM CAM sigmoid
python .\inference.py -m model\CAM_10e-5\epoch_22.pth -t CAM -s 98 -n sigmoid

@REM CAM relu
python .\inference.py -m model\CAM_10e-5\epoch_22.pth -t CAM -s 98 -n relu

@REM ReCAM minmax
python .\inference.py -m model\ReCAM_20e-5_minmax\epoch_12.pth -t ReCAM -s 98 -n minmax

@REM ReCAM sigmoid
python .\inference.py -m model\ReCAM_5e-5_sigmoid\epoch_40.pth -t ReCAM -s 98 -n sigmoid

@REM ReCAM relu hybrid
python .\inference.py -m model\ReCAM_10e-5_relu\epoch_24.pth -t ReCAM -s 98 -n relu

@REM LayerCAM 
python .\inference.py -m model\CAM_10e-5\epoch_22.pth -t LayerCAM -s 98