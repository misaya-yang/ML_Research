"""Development-only line-localization helpers. Candidate code never execs on host."""
import ast
import io
import tokenize

def traced_worker(worker):
    target = '        exec(compile(test,"<test>","exec",optimize=0),trusted)'
    assert worker.count(target) == 1
    new = '''        def tracer(frame,event,arg):
            if frame.f_code.co_filename == "<candidate>" and event == "line":
                if 1 <= frame.f_lineno <= 1024:
                    line_hits.pop(frame.f_lineno, None)
                    line_hits[frame.f_lineno] = True
            return tracer
        settrace = sys.settrace
        settrace(tracer)
        try:
            exec(compile(test,"<test>","exec",optimize=0),trusted)
        finally:
            settrace(None)'''
    worker = worker.replace(target, new).replace('code,setup,test=json.load(sys.stdin)', 'code,setup,test=json.load(sys.stdin)\nline_hits={}')
    last='print(json.dumps([status,reason[:120]]))'
    assert worker.count(last) == 1
    return worker.replace(last,'print(json.dumps([status,reason[:120],list(line_hits)]))')

def operation_lines(code):
    """Executable statements, operators, arguments, literals; never doc/comments."""
    try:
        t=ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    doc = set()
    for node in ast.walk(t):
        body = getattr(node, 'body', None)
        if isinstance(body,list) and body and isinstance(body[0],ast.Expr):
            v=body[0].value
            if isinstance(v,ast.Constant) and isinstance(v.value,str):
                doc.update(range(v.lineno,v.end_lineno+1))
    return {n.lineno for n in ast.walk(t)
            if isinstance(n,(ast.stmt,ast.Call,ast.BinOp,ast.BoolOp,ast.UnaryOp,ast.Compare,ast.Constant))
            and hasattr(n,'lineno')} - doc

def operation_point(code, line, column):
    if line not in operation_lines(code):
        return False
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.COMMENT,tokenize.STRING) and t.start <= (line,column) < t.end:
                return False
    except (tokenize.TokenError,IndentationError):
        return False
    return True
