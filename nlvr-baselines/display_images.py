#!/usr/bin/python


import cgi
import json
import string
import sys

import definitions


def canvas_info_str(id):
    return "<tr><td><b>" + id + "</b><br /><canvas id=\"" + id + "\" width = \"400\" height = \"100\"></canvas></td></tr>"


print("<script src=\"image_gen.js\"></script><html><body><center>")

infile_lines = open(definitions.TRAIN_JSON, "r").read()
infile_images = infile_lines.split("\n")

# This creates canvas elements for each image, with the utterance and judgment
# below.
count = 0
images_worlds = []
img_desc_def = []
for image in infile_images:
    if image:
        json_obj = json.loads(image)
        utterance = json_obj["sentence"]
        evaluation = json_obj["label"]
        display_image = json_obj["structured_rep"]

        print
        ("<table>")

        print
        canvas_info_str("image_" + str(count))
        print
        ("</table>")
        print
        ("<br />")
        print
        "<b>" + utterance + "</b><br />"
        print
        (evaluation)
        print
        ("<br /><br />")

        worlds = ["image_" + str(count)]
        images_worlds.append(worlds)

        img_desc_def.append((utterance, display_image))
        count += 1

print("</center></body></html>")

# Populate the canvas elements by running the script to render them.
print("<script type = \"text/javascript\">")

index = 0
for image in images_worlds:
    for world in image:
        print
        ("var " + world + " = document.getElementById(\"" + world + "\").getContext(\"2d\")")

        desc_formatted = json.dumps(img_desc_def[index][1])
        print
        ("worlds_json = JSON.parse(" + desc_formatted + ")")
        print
        ("draw_objects(" + world + ", worlds_json)")
    index += 1

print("</script>")