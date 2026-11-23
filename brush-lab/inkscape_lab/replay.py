"""Recreate paper and every local stochastic mark using its recorded seed."""
import json
import gzip
from pathlib import Path
from .brush import InkCanvas


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def replay(ledger,output=None,limit=None):
    """Replay a complete landscape ledger (or its first ``limit`` applications).

    Missing physical state is an error, rather than silently receiving today's
    brush defaults. ``fade_bottom`` is optional for ledgers made before fading
    was introduced; its absence means zero. Exact pixels are guaranteed only
    in the same numerical environment and with the same renderer version.
    """
    if limit is not None:
        _integer(limit, 'limit')
    path=Path(ledger)
    data=json.loads(gzip.decompress(path.read_bytes()).decode('utf8') if path.suffix=='.gz' else path.read_text(encoding='utf8'))
    if not isinstance(data, dict):
        raise ValueError('ledger must be an object containing width, height, recipe and strokes')
    if not isinstance(data.get('recipe'), dict) or 'seed' not in data['recipe']:
        raise ValueError('ledger recipe must contain the paper seed')
    width=_integer(data.get('width'), 'width', 1)
    height=_integer(data.get('height'), 'height', 1)
    seed=_integer(data['recipe']['seed'], 'recipe.seed')
    if not isinstance(data.get('strokes'), list):
        raise ValueError('ledger strokes must be a list')
    paper_color=data.get('paper_color', (244, 240, 228))
    if (not isinstance(paper_color, (list, tuple)) or len(paper_color)!=3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not 0 <= v <= 255 for v in paper_color)):
        raise ValueError('paper_color must contain three finite channels in [0, 255]')
    c=InkCanvas(width,height,seed=seed,paper_color=paper_color)
    for index,source in enumerate(data['strokes'][:limit]):
        if not isinstance(source, dict):
            raise ValueError(f'application {index} must be an object')
        kind=source.get('kind')
        if kind in ('painting_stroke','drawing_stroke'):
            required={'program','brush','ink','paper','feedback','initial_state','context','seed','version','driver_hash'}
            if kind=='drawing_stroke':
                required.add('final_state')
            missing=required-source.keys()
            if missing:
                raise ValueError(f'application {index} missing {kind} fields: {", ".join(sorted(missing))}')
            from .painting import paint_mark, FEEDBACK_DEFAULTS, VERSION
            from .gesture import compile_gesture
            execute, compile_program = paint_mark, compile_gesture
            if kind=='drawing_stroke':
                from .drawing import draw_mark, compile_drawing, VERSION
                execute, compile_program = draw_mark, compile_drawing
            from .response import BRUSH_DEFAULTS, INK_DEFAULTS, PAPER_DEFAULTS
            if source['version']!=VERSION:
                raise ValueError(f'application {index} has unsupported {kind} version')
            _integer(source['seed'],f'application {index} seed')
            for group,defaults in (('brush',BRUSH_DEFAULTS),('ink',INK_DEFAULTS),('paper',PAPER_DEFAULTS),('feedback',FEEDBACK_DEFAULTS)):
                if not isinstance(source[group],dict) or set(source[group])!=set(defaults):
                    raise ValueError(f'application {index} must contain all resolved {group} parameters')
            if not isinstance(source['initial_state'],dict):
                raise ValueError(f'application {index} must contain resolved initial_state')
            if compile_program(source['program'])['driver_hash']!=source['driver_hash']:
                raise ValueError(f'application {index} driver hash does not match its program')
            record=execute(c,source['program'],brush=source['brush'],ink=source['ink'],paper=source['paper'],
                           feedback=source['feedback'],initial_state=source['initial_state'],context=source['context'],seed=source['seed'])
            if kind=='drawing_stroke' and record['final_state']!=source['final_state']:
                raise ValueError(f'application {index} drawing final_state does not match replayed state')
            continue
        if kind=='gesture_stroke':
            required={'program','brush','ink','paper','seed','version','driver_hash'}
            missing=required-source.keys()
            if missing:
                raise ValueError(f'application {index} missing gesture fields: {", ".join(sorted(missing))}')
            if source['version']!='gesture-response-0.3.0':
                raise ValueError(f'application {index} has unsupported gesture version')
            _integer(source['seed'],f'application {index} seed')
            from .gesture import gesture_stroke, compile_gesture
            from .response import BRUSH_DEFAULTS, INK_DEFAULTS, PAPER_DEFAULTS
            for group, defaults in (('brush',BRUSH_DEFAULTS),('ink',INK_DEFAULTS),('paper',PAPER_DEFAULTS)):
                if not isinstance(source[group],dict) or set(source[group])!=set(defaults):
                    raise ValueError(f'application {index} must contain all resolved {group} parameters')
            if compile_gesture(source['program'])['driver_hash']!=source['driver_hash']:
                raise ValueError(f'application {index} driver hash does not match its program')
            gesture_stroke(c,source['program'],brush=source['brush'],ink=source['ink'],
                           paper=source['paper'],seed=source['seed'])
            continue
        if kind=='dynamic_stroke':
            if not isinstance(source.get('spec'),dict):
                raise ValueError(f'application {index} dynamic_stroke must contain its spec')
            if 'seed' not in source['spec']:
                raise ValueError(f'application {index} dynamic_stroke spec must contain seed')
            _integer(source['spec']['seed'],f'application {index} seed')
            from .dynamic import dynamic_stroke
            dynamic_stroke(c,source['spec'])
            continue
        if kind not in ('stroke','wash'):
            raise ValueError(f'application {index} has unknown or missing kind: {kind!r}')
        required={'points','ink','water','tag','seed'}
        if kind=='stroke': required.update(('width','dry','pressure','tilt'))
        missing=required-source.keys()
        if missing:
            raise ValueError(f'application {index} missing fields: {", ".join(sorted(missing))}')
        _integer(source['seed'], f'application {index} seed')
        entry=dict(source); entry.pop('kind'); entry.pop('id',None)
        points=entry.pop('points')
        if kind=='wash':
            c.wash(points,ink=entry['ink'],water=entry['water'],tag=entry['tag'],seed=entry['seed'],
                   fade_bottom=entry.get('fade_bottom',0.))
        else:
            c.stroke(points,**entry)
    if output: c.save(output)
    return c
