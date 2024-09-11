import os
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
from fileutils import parse_filename, calculate_average_result, save_average_results_to_csv, \
    load_average_results_from_csv, plot_filtered_objects_over_time

def process_folder(source):
    model = YOLO(r"E:\Dissertation\yolov8-main\runs\train\yolov8s_img360_iou5_sgd\weights\best.pt")  # select your model.pt path
    results = model.predict(
        source=source,
        imgsz=1440,  # WARNING imgsz must be multiple of max stride 32
        project='runs/detect',
        # batch=16,  # 批处理大小
        name=os.path.basename(source),  # Use the folder name for experiment name
        save=False,
        line_width=1
    )

    # Create a folder to save results if it doesn't exist
    save_folder = os.path.join(r"E:\Dissertation\average_results0905", os.path.basename(source))
    os.makedirs(save_folder, exist_ok=True)

    data_list = []
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        names = result.names
        image_name = os.path.basename(result.path)
        filename, _ = os.path.splitext(image_name)

        # Initialize target counts dictionary
        target_counts = {name: 0 for name in names.values()}
        for d in reversed(boxes):
            c = int(d.cls)
            target_counts[names[c]] += 1

        parsed_data = parse_filename(image_name, target_counts)
        data_list.append(parsed_data)

    # Calculate average results, now also include 'crop_type'
    group1 = ['variety_type', 'location', 'treatment', 'rep', 'date', 'view_type', 'crop_type']
    average_results = calculate_average_result(data_list, group1)

    # Save the results in a CSV file
    save_path = os.path.join(save_folder, f'{os.path.basename(source)}.csv')
    save_average_results_to_csv(average_results, save_path)

    print(f"Average results saved to {save_path}")

if __name__ == '__main__':
    root_folder = r"E:\Dissertation\full_dataset\winterosr"
    # Loop over each sub-folder in the dataset folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a folder
            process_folder(subfolder_path)
