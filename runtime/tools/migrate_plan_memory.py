import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from seek.policy.plan_memory import migrate_memory


def main():
    parser=argparse.ArgumentParser(description='Migrate a SEEK snapshot')
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    result=migrate_memory(json.loads(args.source.read_text(encoding='utf-8')))

    with args.destination.open('x',encoding='utf-8') as output:
        json.dump(result,output,ensure_ascii=False,allow_nan=False,indent=2)
        output.write('\n')
    print('Evidence retained. Compatible programs retain their interface; older sources require explicit review. Scene intents start empty.')


if __name__=='__main__':main()
