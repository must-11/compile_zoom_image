import argparse
import os
import shutil

from tasks import Compose, CropWindowTask, DetectFaseTask, MakeGroupImgTask


def run(args):
    if os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)
    os.makedirs(args.data_dir)
    tasks = [
        CropWindowTask(args.file_path, args.data_dir),
        DetectFaseTask(args.file_path, args.data_dir, 0.5),
        MakeGroupImgTask(args.file_path, args.out_path, args.body_path,
                         args.background_path, args.data_dir)
    ]
    runner = Compose(tasks)
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("out_path")
    parser.add_argument("--data_dir", default="tmp")
    parser.add_argument("--body_path", default="data/body.png")
    parser.add_argument("--background_path", default="data/todai.jpeg")
    args = parser.parse_args()
    run(args)
