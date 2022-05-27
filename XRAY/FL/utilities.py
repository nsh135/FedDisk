import builtins as __builtin__
from main import logfile

log     =  True
def print(*args, **kwargs):
	__builtin__.print(*args) #log to console

	if log ==True:
		logf= open(logfile, "a")
		for arg in args: logf.write(arg);logf.write("\n")
		logf.close()