# Channels
c1 = 64  # pos,vel,joint embed
c2 = 128  # G,gcn
c3 = 256  # gcn
c4 = 512  # final conv


# NTU (from viewer prespective)
parts_3points_wholebody = [
    # spine
    (1, 0, 16),
    (1, 0, 12),
    (16, 0, 12),
    (20, 1, 0),
    (3, 2, 20),
    # right hand
    (20, 4, 5),
    (4, 5, 6),
    (5, 6, 7),
    (5, 6, 22),
    (6, 7, 21),
    # left hand
    (20, 8, 9),
    (8, 9, 10),
    (9, 10, 11),
    (9, 10, 24),
    (10, 11, 23),
    # right leg
    (0, 12, 13),
    (12, 13, 14),
    (13, 14, 15),
    # left leg
    (0, 16, 17),
    (16, 17, 18),
    (17, 18, 19),
    # upper chest
    (2, 20, 1),
    (2, 20, 8),
    (2, 20, 4),
    (8, 20, 4),
    (1, 20, 8),
    (1, 20, 4),
]

parts_3points_armandhand = [
    # right hand
    (20, 4, 5),
    (4, 5, 6),
    (5, 6, 7),
    (5, 6, 22),
    (6, 7, 21),
    # left hand
    (20, 8, 9),
    (8, 9, 10),
    (9, 10, 11),
    (9, 10, 24),
    (10, 11, 23),
]

parts_2points_interhandandinterfeet = [
    # hand
    (23, 21),
    (24, 22),
    (11, 7),
    (10, 6),
    (9, 5),
    # leg
    (19, 15),
    (18, 14),
    (17, 13),
]
