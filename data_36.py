from roboflow import Roboflow

rf = Roboflow(api_key="Q5EJueInFaJSsaNdPJDc")
project = rf.workspace("trngs-workspace-fm1km").project(
    "smp-s33i6"
)

dataset = project.version(2).download("yolov11")
print(project)
print(dir(project))
