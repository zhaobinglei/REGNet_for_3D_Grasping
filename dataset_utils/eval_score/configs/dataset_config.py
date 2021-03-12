import numpy as np
# baxter
NAME_LIST = sorted(['002_master_chef_can#0', '002_master_chef_can#1',
                    '003_cracker_box#0', '003_cracker_box#1', '003_cracker_box#2', 
                    '004_sugar_box#0', '004_sugar_box#1', '004_sugar_box#2', '004_sugar_box#3',
                    '005_tomato_soup_can#0', '005_tomato_soup_can#1',
                    '006_mustard_bottle#0', '006_mustard_bottle#1', '006_mustard_bottle#2',
                    '007_tuna_fish_can#0', '007_tuna_fish_can#1', '007_tuna_fish_can#2', 
                    '008_pudding_box#0', '008_pudding_box#1', '008_pudding_box#2',
                    '009_gelatin_box#0', '009_gelatin_box#1', '009_gelatin_box#2', '009_gelatin_box#3', 
                    '010_potted_meat_can#0', '010_potted_meat_can#1', '010_potted_meat_can#2', 
                    '011_banana#0', '011_banana#1', #'011_banana#2',
                    '012_strawberry#0', '012_strawberry#1', '012_strawberry#2', 
                    '013_apple#0', '013_apple#1', '013_apple#2', 
                    '014_lemon#0', '014_lemon#1', '014_lemon#2', 
                    '015_peach#0', '015_peach#1', '015_peach#2',
                    '016_pear#0', '016_pear#1', '016_pear#2', 
                    '017_orange#0', '017_orange#1', '017_orange#2',
                    '018_plum#0', '018_plum#1', '018_plum#2', 
                    '019_pitcher_base#0', '019_pitcher_base#1', '019_pitcher_base#2',
                    '021_bleach_cleanser#0', '021_bleach_cleanser#1', '021_bleach_cleanser#2', 
                    #'024_bowl#0', '024_bowl#1', '024_bowl#2',
                    '025_mug#0', #'025_mug#1', '025_mug#2', 
                    '026_sponge#0', '026_sponge#1', '026_sponge#2', '026_sponge#3',
                    #'029_plate#0', '029_plate#1', #'029_plate#2', 
                    '030_fork#0', '030_fork#1', '030_fork#2', '030_fork#3',
                    '031_spoon#0', '031_spoon#1', '031_spoon#2', '031_spoon#3',
                    '032_knife#0', '032_knife#1', '032_knife#2', 
                    '033_spatula#0', '033_spatula#1', '033_spatula#2', 
                    '035_power_drill#0', '035_power_drill#1', #'035_power_drill#2',
                    '036_wood_block#0', '036_wood_block#1',
                    '037_scissors#0', '037_scissors#1', 
                    '038_padlock#0', '038_padlock#1', '038_padlock#2',
                    '040_large_marker#0', '040_large_marker#1', '040_large_marker#2', '040_large_marker#3',
                    '044_flat_screwdriver#0', '044_flat_screwdriver#1', '044_flat_screwdriver#2', 
                    '048_hammer#0', '048_hammer#1', '048_hammer#2',
                    '051_large_clamp#0', '051_large_clamp#1', '051_large_clamp#2', 
                    '056_tennis_ball#0', #'056_tennis_ball#1',
                    '057_racquetball#0', '057_racquetball#1', 
                    '058_golf_ball#0', '058_golf_ball#1',
                    '061_foam_brick#0', '061_foam_brick#1', '061_foam_brick#2', 
                    '063-a_marbles#0', '063-a_marbles#1', '063-a_marbles#2', 
                    '065-a_cups#0', '065-b_cups#0', '065-c_cups#0', 
                    '065-d_cups#0', '065-e_cups#0', '065-f_cups#0', 
                    '065-g_cups#0', #'065-h_cups#0', '065-i_cups#0', '065-j_cups#0',
                    '071_nine_hole_peg_test#0', '071_nine_hole_peg_test#1',
                    '072-b_toy_airplane#0', '072-b_toy_airplane#1', '072-b_toy_airplane#2',
                    '072-c_toy_airplane#0', '072-c_toy_airplane#1', '072-c_toy_airplane#2',
                    '072-d_toy_airplane#0', '072-d_toy_airplane#1', '072-d_toy_airplane#2',
                    '072-e_toy_airplane#0', '072-e_toy_airplane#1', '072-e_toy_airplane#2',
                    '077_rubiks_cube#0', '077_rubiks_cube#1'])

'''
NAME_LIST = sorted(['002_master_chef_can#0',
                    '003_cracker_box#0', '004_sugar_box#0', '004_sugar_box#1',
                    '004_sugar_box#2', '005_tomato_soup_can#0', '005_tomato_soup_can#1',
                    '005_tomato_soup_can#2',
                    '006_mustard_bottle#0', '006_mustard_bottle#1', '006_mustard_bottle#2', '006_mustard_bottle#3',
                    '007_tuna_fish_can#0', '007_tuna_fish_can#1', '007_tuna_fish_can#2', '007_tuna_fish_can#3',
                    '008_pudding_box#0',
                    '008_pudding_box#1', '008_pudding_box#2', '008_pudding_box#3', '009_gelatin_box#0',
                    '009_gelatin_box#1',
                    '009_gelatin_box#2', '009_gelatin_box#3', '010_potted_meat_can#0',
                    '010_potted_meat_can#1',
                    '010_potted_meat_can#2', '011_banana#0', '011_banana#1', '011_banana#2', '011_banana#3',
                    '012_strawberry#0', '012_strawberry#1', '012_strawberry#2', '012_strawberry#3',
                    '013_apple#0',
                    '013_apple#1', '014_lemon#0', '014_lemon#1', '014_lemon#2', '015_peach#0',
                    '015_peach#1',
                    '016_pear#0', '016_pear#1', '017_orange#0',
                    '017_orange#1',
                    '018_plum#0', '018_plum#1', '018_plum#2', '019_pitcher_base#0', '019_pitcher_base#1',
                    '019_pitcher_base#2',
                    '021_bleach_cleanser#0', '021_bleach_cleanser#1', '024_bowl#0', '024_bowl#1', '024_bowl#2',
                    '025_mug#0',
                    '025_mug#1', '025_mug#2', '025_mug#3', '025_mug#4', '026_sponge#0', '026_sponge#1', '026_sponge#2',
                    '026_sponge#3',
                    '026_sponge#4', '029_plate#0', '029_plate#1', '029_plate#2', '033_spatula#0',
                    '033_spatula#1',
                    '033_spatula#2', '035_power_drill#0', '035_power_drill#1', '035_power_drill#2', '035_power_drill#3',
                    '036_wood_block#0', '036_wood_block#1', '038_padlock#0', '038_padlock#1',
                    '038_padlock#2',
                    '040_large_marker#0', '040_large_marker#1', '040_large_marker#2',
                    '040_large_marker#3',
                    '044_flat_screwdriver#0', '044_flat_screwdriver#1', '044_flat_screwdriver#2', '048_hammer#0',
                    '048_hammer#1',
                    '048_hammer#2',
                    '053_mini_soccer_ball#0',
                    '053_mini_soccer_ball#1', '054_softball#0', '054_softball#1',
                    '055_baseball#0',
                    '055_baseball#1', '056_tennis_ball#1',
                    '057_racquetball#0', '057_racquetball#1', '058_golf_ball#0', '058_golf_ball#1',
                    '063-a_marbles#0', '063-a_marbles#1', '063-a_marbles#2', '065-a_cups#0', '065-a_cups#1',
                    '065-b_cups#1', '065-c_cups#1', '065-d_cups#1',
                    '065-f_cups#1', '065-h_cups#0',
                    '071_nine_hole_peg_test#0', '071_nine_hole_peg_test#1', '071_nine_hole_peg_test#2',
                    '072-b_toy_airplane#1', '072-b_toy_airplane#2', '072-c_toy_airplane#1', '072-c_toy_airplane#2',
                    '077_rubiks_cube#0', '077_rubiks_cube#1'])
'''

# DIR_LIST = [(0, 0, 1)]
DIR_LIST = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]

ROUND_FACTOR = 3
TABLE_COLOR = np.array([1, 0.6, 0])


def hash_color(color: tuple):
    assert len(color) == 3
    return int(color[0] * 10 ** (ROUND_FACTOR * 2) + color[1] * 10 ** (ROUND_FACTOR * 1) + color[2])


def hash_original_color(original_color: tuple):
    assert len(original_color) == 3
    color = tuple((np.round(np.array(original_color), ROUND_FACTOR) * 10 ** ROUND_FACTOR).astype(np.int))
    return hash_color(color)


def hash_color_array(color_array: np.ndarray):
    assert color_array.shape[1] == 3
    color = (color_array * 10 ** ROUND_FACTOR).astype(int)
    hash_code = color[:, 0] * (10 ** ROUND_FACTOR * 2) + color[:, 1] * 10 ** (ROUND_FACTOR * 1) + color[:, 2]
    return hash_code


def color_array_to_label(color_array: np.ndarray):
    index = np.rint(color_array[:, 0] * len(NAME_LIST))
    return index


NAME_TO_COLOR = {}
NAME_TO_INDEX = {}
for i, name in enumerate(NAME_LIST):
    percentage_i = i / len(NAME_LIST)
    color_i = np.array([percentage_i, 1 - percentage_i, percentage_i ** 2])
    NAME_TO_COLOR.update({name: color_i})
    NAME_TO_INDEX.update({name: i})

NAME_TO_COLOR.update({'table': TABLE_COLOR})
