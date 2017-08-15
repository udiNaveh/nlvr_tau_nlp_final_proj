from handle_data import *
from logical_forms import *
from structured_rep_utils import *
from definitions import *
from display_images import *

samples, sentences = build_data(read_data(TRAIN_JSON))
print(len(sentences))
print(len(samples))
i=0
for sample in samples:
    #print(sample.sentence)
    if "There are exactly four black objects not touching any edge" in sample.sentence:
        i+=1
        print ('\n'+sample.sentence)
        #print (sample.structured_rep.get_all_boxes())
        print (sample.label==eval('equal_int(4, count(filter(lambda x: not(query_touching_wall(x)) and(equal_color("Black",query_color(x))) , sample.structured_rep.get_all_items())))'))

    if "There is a box with at least one square and at least three triangles" in sample.sentence:
        i+=1
        print ('\n'+sample.sentence)
        #print (sample.structured_rep.get_all_boxes())
        print (sample.label == eval('exist(filter(lambda y: le(1, count(filter_shape("square", y))) and le(3, count(filter_shape("triangle", y))) , sample.structured_rep.get_all_boxes()))'))


    if "There is a tower with yellow base" in sample.sentence:
        i+=1
        print ('\n'+sample.sentence)
        #show_sample(sample)
        #print (sample.structured_rep.get_all_boxes())
        print (sample.label == eval('exist(filter(lambda y: equal_color("Yellow", query_color(unique(filter_location("bottom", y)))) ,sample.structured_rep.get_all_boxes()))'))

    if "There is a black item in every box" in sample.sentence:
        i+=1
        print ('\n'+sample.sentence)
        #print (sample.structured_rep.get_all_boxes())
        print(sample.label == eval('All( lambda y: exist(filter_color("Black", y)),sample.structured_rep.get_all_boxes())'))

        #show_sample(sample)

    if "There are 2 blue circles and 1 blue triangle" in sample.sentence:
        i+=1
        print ('\n'+sample.sentence)
        #print (sample.structured_rep.get_all_boxes())
        print(sample.label == eval('equal_int(2, count(filter_color("#0099ff", filter_shape("circle", sample.structured_rep.get_all_items())))) and equal_int(1, count(filter_color("#0099ff", filter_shape("triangle", sample.structured_rep.get_all_items()))))'))


    if "There is a blue triangle touching the wall with its side" in sample.sentence:#there are 2 label errors here
        i+=1
        print ('\n'+sample.sentence)
        #show_sample(sample)
        #print('label is: ',sample.label)
        #print (sample.structured_rep.get_all_boxes())
        print(sample.label == eval('exist(filter(lambda x: query_touching_right_wall(x) or query_touching_left_wall(x),filter_color("#0099ff", filter_shape("triangle", sample.structured_rep.get_all_items()))))'))


    if "here is one tower with a yellow block above a yellow block" in sample.sentence:#the "above" meaning is not consistent
        i+=1
        print('\n' + sample.sentence)
        show_sample(sample)
        print('label is: ', sample.label)
        print(sample.structured_rep.get_all_boxes())
        print(sample.label == eval('equal_int(1, count(filter(lambda y: exist(filter_color("Yellow", union(relate_sets("above", filter_color("Yellow", y))))),sample.structured_rep.get_all_boxes())))'))

    if "There is a box with seven items and the three black items are the same in shape" in sample.sentence:#the labels are wrong because they treat it like "three items" and not "the three items"
        i += 1
        print('\n' + sample.sentence)
        print(sample.label == eval('exist(filter(lambda y: equal_int(7, count(y)) and (equal_int(3, count(filter_color("Black", y))) and all_same_shape(filter_color("Black", y))), sample.structured_rep.get_all_boxes()))'))

    if "here is exactly one black triangle not touching the edge" in sample.sentence:
        i += 1
        print('\n' + sample.sentence)
        print(sample.label == eval('equal_int(1,count(filter(lambda x: not query_touching_wall(x), filter_color("Black", filter_shape("triangle" , sample.structured_rep.get_all_items())))))'))


print("i =",i)
