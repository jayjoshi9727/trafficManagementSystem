import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import OrderedDict
import statistics as stats



from imageai.Detection import ObjectDetection


def get_priority(vehicle_count):
    print(vehicle_count)
    lane_priority={}
    priority_list = list(vehicle_count.values())
    priority_list.sort(reverse=True)
    print("priorty list ", priority_list)
    itera = 0
    flag = False
    count_diff = priority_list[0] - priority_list[3]
    for count in priority_list:
        if count <= 20:
            itera += 1
            if itera == 4:
                flag=True

    if count_diff <= 20:
        flag = True

    if flag:
        for k, v in vehicle_count.items():
            lane_priority.update({k: 1})
    else:
        for k, v in vehicle_count.items():
            lane_priority.update({k:priority_list.index(vehicle_count.get(k))})
    print("lane priority ",lane_priority)
    return lane_priority


def sort_dic(vehicle_count):
    print(dict(OrderedDict(sorted(vehicle_count.items(), key=lambda t: t[1]))))
    return dict(OrderedDict(sorted(vehicle_count.items(), key=lambda t: t[1])))

def get_timer(vehicle_count):
    lane_delay={}
    # sorted_vehicle_count=sort_dic(vehicle_count)
    lane_count = list(vehicle_count.values())
    lane_count.sort(reverse=True)
    test_cases = [False,False,False]
    itera = 0
    count_diff = lane_count[0] - lane_count[3]
    while(True):
        # case 0 when no of vehicles are less so yellow lights
        for count in lane_count:
            if count <= 20:
                itera += 1
                if itera == 4:
                    test_cases[0] = True
                    break


        #case 1 when there is similar traffic in each lane congestion expense.
        if count_diff <= 20:
            test_cases[1] = True
            break


        #case 2 flucating traffic
        if test_cases[2]==False:
            test_cases[2]=True
            break

    if test_cases[0]:
        for k, v in vehicle_count.items():
            lane_delay.update({k: "Infinity"})
    elif test_cases[1]:
        avg = stats.mean(lane_count)
        delay = int((avg*20)/90)+3
        for k, v in vehicle_count.items():
            lane_delay.update({k: delay})

    elif test_cases[2]:
        for k,v in vehicle_count.items():
            veh_count = vehicle_count.get(k)
            delay = int((veh_count*20)/80)+2
            lane_delay.update({k: delay})

    return lane_delay


def get_img_list():
    cam_img_list = []

    for cam_num in range(1, 5):
        img_list = []
        for img_path in glob.glob('images/camera ' + str(cam_num) + '/*.JPG'):
            img_list.append(img_path)
        cam_img_list.append(img_list)

    return cam_img_list


def plot_img(images, vehicle_count):
    sorted_dic = sort_dic(vehicle_count)
    print(sorted_dic)
    plt.figure()
    plt.suptitle("AI based Traffic Management", fontsize=16)
    for j, image in enumerate(images):
        sub_plt = plt.subplot(2, 2, j + 1)
        sub_plt.set_title('Camera ' + str(j) + ' Count '+str(sorted_dic.get(j)) + ' Priority: '+str(get_priority(sorted_dic).get(j))+'Predicted Split '+str(get_timer(sorted_dic).get(j))+"sec")
        sub_plt.axis('off')
        plt.imshow(mpimg.imread(image))

    plt.show()


def get_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("yolo.h5")
    detector.loadModel()
    return detector


def  main():
    detector = get_detector()
    custom = detector.CustomObjects(
        # bicycle=True,
        car=True,
        motorcycle=True,
        bus=True,
        truck=True
    )

    cam_img_list = get_img_list()
    for img_num in range(0, len(cam_img_list[3])):
        lane_no = 0
        images = []
        vehicle_count = {}

        for cam_num in range(0, 4):
            output_img_path = "output/image-" + str(cam_num) + ".JPG"
            images.append(output_img_path)

            list_of_detection = detector.detectCustomObjectsFromImage(
                custom_objects=custom,
                input_image=cam_img_list[cam_num][img_num],
                output_image_path=output_img_path,
                minimum_percentage_probability=1
            )
            # vehicle_count.append(len(list_of_detection))
            vehicle_count.update({lane_no: len(list_of_detection)})
            lane_no += 1

        plot_img(images, vehicle_count)

        print(get_priority(vehicle_count))
        print(get_timer(vehicle_count))
main()

