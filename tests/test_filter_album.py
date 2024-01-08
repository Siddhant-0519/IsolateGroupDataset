import sys
sys.path.append('/root/workspace/data/IsolateGroupDataset')
# print(sys.path)
import os
import cv2
import numpy as np
import pytest 
from filter_album import CleanPersonData
from skimage.metrics import mean_squared_error 

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURR_DIR, "test_data")

@pytest.fixture
def clean_person_data_instance():
    # root_dir = "/root/workspace/data/IsolateGroupDataset/data/"
    ref_path = os.path.join(TEST_DIR, "ref_img.jpg")
    output_path = os.path.join(CURR_DIR, "test_output")
    cleaner = CleanPersonData(TEST_DIR, ref_path, output_path)
    return cleaner

@pytest.fixture
def test_image_path():
    return os.path.join(TEST_DIR, "test_image_2.jpg")

def calculate_rmse(img1 , img2):
    
    image1 = img1.astype(np.float32)
    image2 = img2.astype(np.float32)

    mse = mean_squared_error(image1, image2)
    rmse = np.sqrt(mse)

    return rmse

def test_process_file(clean_person_data_instance, test_image_path):

    clean_person_data_instance.process_file(test_image_path)

    test_output_path = os.path.join(clean_person_data_instance.output_path, "test_image_2-ref-ref_img.jpg")
    sample_image_path = os.path.join(TEST_DIR, "sample_image_2.jpg")
    
    # print(test_output_path)
    # assert os.path.isfile(test_output_path), f"Output file {output_path} does not exist"
    
    test_output = cv2.imread(test_output_path)
    sample_output = cv2.imread(sample_image_path)
    
    rmse = calculate_rmse(test_output , sample_output)
    

    assert rmse <= 0.5, "Different output than expected"