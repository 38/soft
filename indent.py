# The reason why I write such a script is I think the best way for the indentation is 
# mixing spaces with tabs. A lot of people suggest that we should totally use space 
# However, this increase the size of the source code and makes other unconforatble 
# with the "hard indentation". When I have a smaller screen, I need the tabstop become
# smaller, however the space approach fail to do this.
# So I think the good way for indentation is use tabs for indentation, use space for 
# alignment. 
# This tool is written for the reason above
import sys
import re
import os

def format(fp):
	comment = False
	string  = False
	escape = False
	idlevel = 0
	recent = "  "
	result = ""
	for line in fp:
		spaces = 0
		for ch in line:
			if ch == ' ': spaces += 1
			elif ch == '\t': spaces += 4
			else: break
		level = idlevel - (len(line.strip()) != 0 and line.strip()[0] == '}')
		spaces -= level * 4
		if spaces < 0: spaces = 0
		header = '\t'*level + ' '*spaces  
		for ch in line:
			recent = (recent + ch)[1:]
			if not comment and not string:
				if recent == "//": break
				elif recent == "/*": comment = True 
				elif recent[1] == "\"": string = True
			elif comment:
				if recent == "*/": comment = False
			elif string:
				if not escape and recent[1] == "\\": escape = True
				elif escape: escape = False
				elif recent[1] == '"': string = False
			if not comment and not string:
				if(ch == '{'): idlevel += 1
				if(ch == '}'): idlevel -= 1
		result += header + line.strip() + "\n"
	fp.close()
	return result
				

fn=r'.*\.(c|h|cpp|cxx|hpp)$'
ts = 4

for root, _, files in os.walk("."):
	for f in files:
		if not re.match(fn,f): continue
		path="%s/%s"%(root,f)
		result = format(file(path))
		f = file(path, "w")
		f.write(result)
		f.close()


		
