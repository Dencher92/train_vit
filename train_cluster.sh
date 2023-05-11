# the only thing you need is to change CONFIG_NAME env var
PYTHON_EXECUTABLE=/opt/conda/envs/main/bin/python
MNT_DIR_PATH=/mnt/home/divashkov
CONFIG_DIR=/mnt/home/divashkov/vit_t/
REPO_NAME=vit_t
SCRIPT_NAME=train_vit_2.py
CONFIG_NAME=train_2

cd $MNT_DIR_PATH/$REPO_NAME/;
CONFIG_DIR=$CONFIG_DIR \
CONFIG_NAME=$CONFIG_NAME \
$PYTHON_EXECUTABLE $SCRIPT_NAME \
++mnt_dir_path=$MNT_DIR_PATH \
++is_cluster=True \
"$@"
