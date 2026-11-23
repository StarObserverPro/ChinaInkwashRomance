"""Run with python -m inkscape_lab --help from the unpacked project directory."""
import argparse
from pathlib import Path
from .landscape import Recipe,render

def main():
    parser=argparse.ArgumentParser(description='Ink Landscape Lab / 水墨山水笔触实验')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('render',help='render a reproducible landscape and stroke ledger')
    p.add_argument('--round',type=int,choices=[0,1,2,3],default=3)
    p.add_argument('--seed',type=int,default=731)
    p.add_argument('--width',type=int,default=1200)
    p.add_argument('--height',type=int,default=1440)
    p.add_argument('--output',type=Path,default=Path('examples/landscape.png'))
    p=sub.add_parser('replay',help='exact replay from a recorded stroke ledger')
    p.add_argument('ledger',type=Path)
    p.add_argument('--output',type=Path,default=Path('examples/replay.png'))
    p.add_argument('--limit',type=int,default=None)
    sub.add_parser('studies',help='brush, form and physics experiment plates')
    p=sub.add_parser('dynamics',help='dynamic stroke, contact boundary and cun experiments')
    p.add_argument('--revision',type=int,choices=[0,1],default=0)
    p.add_argument('--output',type=Path,default=Path('examples'))
    p=sub.add_parser('painting',help='paint shared rock faces with role-specific boundaries and live density feedback')
    p.add_argument('--revision',type=int,choices=[0,1,2],default=1)
    p.add_argument('--output',type=Path,default=Path('examples'))
    p=sub.add_parser('drawing',help='draw one stone with exactly three contacts and discrete form actions')
    p.add_argument('--revision',type=int,choices=[0,1,2],default=2)
    p.add_argument('--output',type=Path,default=Path('examples'))
    p=sub.add_parser('drawing-gallery',help='build the small offline three-stroke drawing gallery')
    p.add_argument('--examples',type=Path,default=Path('examples'))
    p.add_argument('--output',type=Path,default=Path('drawing_gallery.html'))
    p=sub.add_parser('gestures',help='discrete action programs and independent material response studies')
    p.add_argument('--revision',type=int,choices=[0,1],default=0)
    p.add_argument('--output',type=Path,default=Path('examples'))
    p=sub.add_parser('gallery',help='build the self-contained offline gallery')
    p.add_argument('--examples',type=Path,default=Path('examples'))
    p.add_argument('--output',type=Path,default=Path('gallery.html'))
    args=parser.parse_args()
    if args.command=='render':
        if args.width<200 or args.height<200 or args.width>6000 or args.height>7200: parser.error('width/height outside supported 200..6000 / 200..7200 range')
        c=render(Recipe(round=args.round,seed=args.seed,width=args.width,height=args.height),args.output)
        print(f'{args.output}: {len(c.strokes)} recorded applications')
    elif args.command=='replay':
        from .replay import replay
        replay(args.ledger,args.output,args.limit)
        print(args.output)
    elif args.command=='studies':
        from .studies import make_studies
        make_studies(Path('examples'))
    elif args.command=='dynamics':
        from .dynamic_studies import make_dynamic_studies
        make_dynamic_studies(args.output,args.revision)
    elif args.command=='gallery':
        from .gallery import build_gallery
        build_gallery(args.examples,args.output)
        print(args.output)
    elif args.command=='gestures':
        from .drive_studies import make_drive_studies
        make_drive_studies(args.output,args.revision)
    elif args.command=='painting':
        from .painting_studies import make_painting_studies
        make_painting_studies(args.output,args.revision)
    elif args.command=='drawing':
        from .drawing_studies import make_drawing_studies
        make_drawing_studies(args.output,args.revision)
    elif args.command=='drawing-gallery':
        from .drawing_gallery import build_drawing_gallery
        print(build_drawing_gallery(args.examples,args.output))

if __name__=='__main__': main()
