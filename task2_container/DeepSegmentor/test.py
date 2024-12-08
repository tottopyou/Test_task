import os
import logging
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/app/logs/model_test.log")]
)

if __name__ == '__main__':
    opt = TestOptions().parse()  # Get test options

    # Hard-code some parameters for test
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # Website directory
    web_dir = './results'
    logging.info(f"Results will be saved to: {web_dir}")
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Dataset root path: {opt.dataroot}")
    webpage = html.HTML(web_dir, 'Experiment Results')

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        logging.info(f"Processing image {i}...")
        if not data:
            logging.warning("No data returned for this iteration.")

        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if visuals:
            logging.info(f"Generated visuals for: {img_path}")
        else:
            logging.warning(f"No visuals for: {img_path}")

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
    logging.info("HTML result saved successfully.")
