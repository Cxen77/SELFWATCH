"""Bounded subprocess entry point: detect an explicitly requested local camera."""
import argparse
import json
import sys
import cv2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    if not 0<=args.index<=3:raise ValueError('Camera index outside supported range')
    cap=cv2.VideoCapture(args.index,cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        ok,image=cap.read()
        print(json.dumps(dict(index=args.index,available=bool(ok and image is not None),
            width=image.shape[1] if ok else None,height=image.shape[0] if ok else None)))
    finally:cap.release()


if __name__=='__main__':main()
