import os

for c in range(30):
    for s in range(30):
        style_index = str(s)
        content_index = str(c)
        command = "python scripts/adversary_synthesis.py --style_image classic-konkle-buildings --style_index " + style_index + " --content_image classic-konkle-faces --content_index " + content_index + " --content_weight 10 --num_steps 800"
        print(command)
        os.system(command) 

        
