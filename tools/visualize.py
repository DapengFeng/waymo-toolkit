import argparse

from waymo_extractor.viewer import Viewer2D, Viewer3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="provide source path to visualize", type=str
    )
    parser.add_argument("--image", action="store_true", help="whether to show image")
    parser.add_argument("--laser", action="store_true", help="whether to show laser")
    parser.add_argument("-s", "--step", default=1, help="the step for visualize the data", type=int)
    parser.add_argument(
        "-c", "--camera", default=0, help="the camera for visualize the data", type=int
    )
    parser.add_argument("--box2d", action="store_true", help="draw the projected 2d box")
    parser.add_argument("--box3d", action="store_true", help="draw the projected 3d box")
    parser.add_argument("--project", action="store_true", help="draw the projected laser on image")
    args = parser.parse_args()

    if args.image:
        viewer2d = Viewer2D(args)
        viewer2d.display(camera=args.camera, step=args.step)

    if args.laser:
        viewer3d = Viewer3D(args)
        viewer3d.display(step=args.step)


if __name__ == "__main__":
    main()
