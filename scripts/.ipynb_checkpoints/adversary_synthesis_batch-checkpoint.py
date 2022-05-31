import os

# for c in range(3,6):
#     for s in range(3):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image face --style_index " + style_index + " --content_image scene-konkle-3-3-japaneseroom --content_index " + content_index + " --content_weight 10 --num_steps 1500"
#         os.system(command)

# for c in range(7,10):
#     for s in range(3):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image face --style_index " + style_index + " --content_image scene-konkle-4-1-loft --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command)        

# for c in range(0,4):
#     for s in range(3):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image face --style_index " + style_index + " --content_image scene-konkle-4-3-lobby --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command) 

# for c in range(30):
#     for s in range(30):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image classic-konkle-faces --style_index " + style_index + " --content_image classic-konkle-buildings --content_index " + content_index + " --content_weight 10 --num_steps 800"
#         print(command)
#         os.system(command) 

        
for c in range(30):
    for s in range(30):
        style_index = str(s)
        content_index = str(c)
        command = "python scripts/adversary_synthesis.py --style_image classic-konkle-buildings --style_index " + style_index + " --content_image classic-konkle-faces --content_index " + content_index + " --content_weight 10 --num_steps 800"
        print(command)
        os.system(command) 

        
# for c in range(5):
#     for s in range(c*2, c*2+5):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image scene --style_index " + style_index + " --content_image face --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command)

# for c in range(5):
#     for s in range(c*2, c*2+5):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image scene --style_index " + style_index + " --content_image object --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command)

# for c in range(5):
#     for s in range(c*2, c*2+5):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image object --style_index " + style_index + " --content_image face --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command)
