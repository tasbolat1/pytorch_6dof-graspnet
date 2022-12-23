#!/bin/bash
python gpd_preprocess.py --cat mug --idx 2 
python gpd_preprocess.py --cat mug --idx 8  
python gpd_preprocess.py --cat mug --idx 14  

python gpd_preprocess.py --cat bottle --idx 3  
python gpd_preprocess.py --cat bottle --idx 12  
python gpd_preprocess.py --cat bottle --idx 19  

python gpd_preprocess.py --cat box --idx 14  
python gpd_preprocess.py --cat box --idx 17  

python gpd_preprocess.py --cat bowl --idx 1  
python gpd_preprocess.py --cat bowl --idx 16  

python gpd_preprocess.py --cat cylinder --idx 2  
python gpd_preprocess.py --cat cylinder --idx 11  

python gpd_preprocess.py --cat pan --idx 3  
python gpd_preprocess.py --cat pan --idx 6  

python gpd_preprocess.py --cat scissor --idx 4  
python gpd_preprocess.py --cat scissor --idx 7  

python gpd_preprocess.py --cat fork --idx 1  
python gpd_preprocess.py --cat fork --idx 11  

# hammer002 has problem
python gpd_preprocess.py --cat hammer --idx 2  
python gpd_preprocess.py --cat hammer --idx 15  

python gpd_preprocess.py --cat spatula --idx 1  
python gpd_preprocess.py --cat spatula --idx 14  