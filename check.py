from ai2thor.platform import CloudRendering
from ai2thor.controller import Controller
import json
import prior
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='-1'

dataset = prior.load_dataset("procthor-10k")
tests = dataset["test"]
print(len(list(tests)))
for house in tests:
    
    env = Controller(scene=house, platform=CloudRendering)
    print(env.last_action)
    env.stop()
    break


