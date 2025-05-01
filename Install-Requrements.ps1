#pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu128

# requrements.txtの -e . をコメントアウトする必要あり
#pip install --upgrade -r requirements.txt
#pip install xformers==0.0.29.post2

uv add torch torchvision --index https://download.pytorch.org/whl/cu128

# requrements.txtの -e . をコメントアウトする必要あり
uv pip install --upgrade -r .\requirements.txt
uv add xformers --index https://download.pytorch.org/whl/cu128
uv add numpy==1.26.3
