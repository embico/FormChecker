# reminder - install packages by:
# view->command pallete -> Terminal: Create New Terminal
# in new terminal run: python3 -m pip install <lib>
# before commiting run: python3 -m  pipreqs.pipreqs .

import image_processor

print("main - starting")

proc = image_processor.Processor("vids/squat1.MOV")
proc.print()
proc.read()