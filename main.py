# reminder - install packages by:
# view->command pallete -> Terminal: Create New Terminal
# in new terminal run: pip install <lib>
# before commiting run: pip freeze > requirements.txt

import image_processor

print("main - starting")

proc = image_processor.Processor("vids/squat1.MOV")
proc.print()
proc.read()