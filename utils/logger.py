import sys, socket
from string import Template
from datetime import datetime

class logger():
	def __init__(self, fn, mute=False, max_hold=0 , enable_servername=True, enable_timestamp=True):
		self.console = getattr(sys.stdout, "console", sys.stdout)
		self.logfile = open( str(fn), "a" )
		self.servername = ("[" + socket.gethostname().split(".")[0] + "]") if enable_servername else ""
		self.enable_timestamp = enable_timestamp
		self.max = max_hold
		self.mute = mute
		self.counter = 0

		self.was_linebreak = True

	def __del__(self):
		# close logger
		self.logfile.write("\n")
		self.console.write("\n")
		self.flush()		

	def write(self, s):

		s = str(s)

		if self.was_linebreak:
			s = "\n" + s
			self.was_linebreak = False
		if s[-1] == "\n":
			s = s[:-1]
			self.was_linebreak = True

		
		templ = Template(s.replace("\n", "\n${prefix} "))

		if not(self.mute):
			self.console.write( templ.substitute(prefix=self.servername) )
			self.console.flush()

		if len(s) > 0:
			self.logfile.write( templ.substitute(prefix=self.servername + ((str(datetime.now()) + " |") if self.enable_timestamp else "")) )

			self.counter += 1
			if self.counter >= self.max:
				self.flush()

	def flush(self):
		self.counter = 0
		self.logfile.flush()
		self.console.flush()
