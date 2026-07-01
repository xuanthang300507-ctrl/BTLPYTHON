from roboflow import Roboflow

rf = Roboflow(api_key="WjYeXfKJXkiBln8qiYtm")
project = rf.workspace("thang-nguyen-fcp4i").project(
    "vietnamese-license-plate-tptd0-np3du"
)

dataset = project.version(1).download("yolov11")
print(project)
print(dir(project))