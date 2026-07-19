import os
import argparse


def build_args():
    parser = argparse.ArgumentParser("This script is used to Source-free Universal Domain Adaptation")

    parser.add_argument("--dataset", type=str, default="Office", help="Office, OfficeHome, VisDA, DomainNet")
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--llm_model', default="gpt-4-turbo", type=str, choices=['gpt-3.5-turbo-16k', 'gpt-4-turbo'])
    parser.add_argument('--url', type=str, default="https://xiaoai.plus/v1")
    parser.add_argument('--api_key', type=str, default="sk-40LQIyWIzbonUqeKcHokPvsW4mhrGI7nhqtoBTc15dXHRILh")

    parser.add_argument('--t', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--index', type=int, default=1)

    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--epochs", default=10, type=int)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpu", default='7', type=int)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--nbr", default=5, type=int)
    parser.add_argument("--w_0", default=0.55, type=float)

    parser.add_argument("--source_train_type", default="smooth", type=str, help="vanilla, smooth")
    parser.add_argument("--target_label_type", default="OPDA", type=str, help="PDA, OSDA, OPDA, CLDA")
    parser.add_argument("--target_private_class_num", default=None, type=int)
    parser.add_argument("--note", default="2024")
    
    # cv计算百分比
    parser.add_argument("--per", default=3, type=int)

    args = parser.parse_args()
    
    args.per_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    '''
    assume classes across domains are the same.
    [0 1 ............................................................................ N - 1]
    |---- common classes --||---- source private classes --||---- target private classes --|

    |-------------------------------------------------|
    |                DATASET PARTITION                |
    |-------------------------------------------------|
    |DATASET    |  class split(com/sou_pri/tar_pri)   |
    |-------------------------------------------------|
    |DATASET    |    PDA    |    OSDA    | OPDA/UniDA |
    |-------------------------------------------------|
    |Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
    |-------------------------------------------------|
    |OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
    |-------------------------------------------------|
    |VisDA-C    |   6/6/0   |   6/0/6    |   6/3/3    |
    |-------------------------------------------------|  
    |DomainNet  |           |            | 150/50/145 |
    |-------------------------------------------------|
    '''

    if args.dataset == "Office":
        domain_list = ['Amazon', 'Dslr', 'Webcam']
        args.source_data_dir = os.path.join("./data/Office", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/Office", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/Office", item) for item in args.target_domain_list]

        labels = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer',
                  'monitor', 'mouse', 'mug', 'projector', 'bike_helmet', 'bookcase', 'bottle', 'desk_chair',
                  'desk_lamp', 'desktop_computer', 'file_cabinet', 'letter_tray', 'mobile_phone', 'paper_notebook',
                  'pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
                  'tape_dispenser', 'trash_can']

        args.shared_class_num = 10

        if args.target_label_type == "PDA":
            args.source_private_class_num = 21
            args.target_private_class_num = 0

        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 10

        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 10
            if args.target_private_class_num is None:
                args.target_private_class_num = 11

        elif args.target_label_type == "CLDA":
            args.shared_class_num = 31
            args.source_private_class_num = 0
            args.target_private_class_num = 0

        else:
            raise NotImplementedError("Unknown target label type specified")

    elif args.dataset == "OfficeHome":
        domain_list = ['Art', 'Clipart', 'Product', 'Realworld']
        args.source_data_dir = os.path.join("./data/OfficeHome", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/OfficeHome", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(4) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/OfficeHome", item) for item in args.target_domain_list]

        labels = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket',
                  'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch',
                  'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops',
                  'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives',
                  'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
                  'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator',
                  'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon',
                  'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

        if args.target_label_type == "PDA":
            args.shared_class_num = 25
            args.source_private_class_num = 40
            args.target_private_class_num = 0

        elif args.target_label_type == "OSDA":
            args.shared_class_num = 25
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 40

        elif args.target_label_type == "OPDA":
            args.shared_class_num = 10
            args.source_private_class_num = 5
            if args.target_private_class_num is None:
                args.target_private_class_num = 50

        elif args.target_label_type == "CLDA":
            args.shared_class_num = 65
            args.source_private_class_num = 0
            args.target_private_class_num = 0
        else:
            raise NotImplementedError("Unknown target label type specified")

    elif args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.target_domain_list = ["validataion"]
        args.target_domain_dir_list = [args.target_data_dir]

        labels = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
                  'person', 'plant', 'skateboard', 'train', 'truck']

        args.shared_class_num = 6
        if args.target_label_type == "PDA":
            args.source_private_class_num = 6
            args.target_private_class_num = 0

        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            args.target_private_class_num = 6

        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 3
            args.target_private_class_num = 3

        elif args.target_label_type == "CLDA":
            args.shared_class_num = 12
            args.source_private_class_num = 0
            args.target_private_class_num = 0

        else:
            raise NotImplementedError("Unknown target label type specified", args.target_label_type)

    elif args.dataset == "DomainNet":
        domain_list = ["Painting", "Real", "Sketch"]
        args.source_data_dir = os.path.join("./data/DomainNet", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/DomainNet", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/DomainNet", item) for item in args.target_domain_list]
        args.embed_feat_dim = 512  # considering that DomainNet involves more than 256 categories.

        labels = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier',
                  'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
                  'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball',
                  'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed',
                  'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry',
                  'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread',
                  'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus',
                  'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle',
                  'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello',
                  'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
                  'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile',
                  'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin',
                  'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow',
                  'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence',
                  'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight',
                  'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
                  'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar',
                  'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet',
                  'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog',
                  'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail',
                  'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg',
                  'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
                  'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
                  'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache',
                  'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion',
                  'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip',
                  'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
                  'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car',
                  'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon',
                  'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river',
                  'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus',
                  'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe',
                  'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face',
                  'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider',
                  'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
                  'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean',
                  'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt',
                  'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger',
                  'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor',
                  'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella',
                  'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale',
                  'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

        args.shared_class_num = 150
        if args.target_label_type == "OPDA":
            args.source_private_class_num = 50
            args.target_private_class_num = 145
        else:
            raise NotImplementedError("Unknown target label type specified")

    else:
        raise NotImplementedError("Unknown target dataset")

    args.source_class_num = args.shared_class_num + args.source_private_class_num
    args.target_class_num = args.shared_class_num + args.target_private_class_num
    args.class_num = args.source_class_num

    args.L = int(args.class_num * args.alpha)
    args.test_labels = labels[: args.shared_class_num + args.source_private_class_num]

    args.source_class_list = [i for i in range(args.source_class_num)]
    args.target_class_list = [i for i in range(args.shared_class_num)]
    if args.target_private_class_num > 0:
        args.target_class_list.append(args.source_class_num)

    return args
