import os
import time as tt
import math as m
import random as rd
import numpy as np
from numpy.lib import scimath

tmp_file = "/tmp/.tmp_file"
img = ""

def get_ecr_size():
    tmp_file = "/tmp/."+ str(rd.randint(0, 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)) + ""
    cmd = "rm -rf " + tmp_file + "; touch " + tmp_file + "; tput cols > " + tmp_file + "; tput lines >> " + tmp_file + ";"
    os.system(cmd)
    f = open(tmp_file, "r")
    size = [int(f.readline()), int(f.readline())]
    f.close
    cmd = "rm -rf " + tmp_file + ";"
    os.system(cmd)
    return (size)

def get_ecr_size_vec():
    size = get_ecr_size()
    return (np.array(size))

def get_ecr_size2():
    size = get_ecr_size()
    return ({'x':size[0], 'y':size[1]})
    

def get_col(val):
    val = val if ((val >= 0.0) and (val <= 1.1)) else (0.0 if (val < 0.0) else 1.0)
    color = [' ', '.', '-', ':', '*', '0', '@', '#']
    id = int(val * (len(color) - 1))
    return (color[id])

#def print_line(img2, size, p1, p2, c1, c2):
#    global img
#
#    diff = [(p2[0] - p1[0]), (p2[1] - p1[1])]
#    col_dif = [(c2[0] - c1[0]), (c2[1] - c1[1]), (c2[2] - c1[2])]
#   # len_line = m.sqrt(diff[0]**2 + diff[1]**2)
#    id_max = 0 if (abs(diff[0]) >= abs(diff[1])) else 1
#    id_min = 1 - id_max
#    ratio = diff[id_min] / diff[id_max]
#    nb_pix = diff[id_max]
#    pos = [0, 0]
#    print("id_min:", id_min, "id_max:", id_max)
#    for i in range(nb_pix):
#    pos[id_max] = 0#float(p1[id_max] + i)
#    pos[id_min] = p1[id_min] + float(i * ratio) / (nb_pix)
#    img[int(pos[0] + pos[1] * size[0])] = "9"#get_col(c1)

#################################################


###### DATA ######
# {position + axe} camera
# liste des position des point
# liste des liens des point
# ==> liste des segment

class Vector:

    def __init__(self, x = 0.0, y = 0.0, z = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def sub(a, b):
        return (Vector(a.x - b.x, a.y - b.y, a.z - b.z))
    
    def add(a, b):
        return (Vector(a.x + b.x, a.y + b.y, a.z + b.z))
    def scalar_product(vec, coef):
        return (Vector(vec.x * coef, vec.y * coef. vec.z * coef))
    def dot_product(v1, v2):
        return (v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
    def normalize(vec):
        return (scalar_product(vec, 1.0 / m.sqrt(dot_product(vec, vec))))

class Segment:

    def __init__(self, pt1 = Vector(0., 0., 0.), pt2 = Vector(1., 1., 1.)):
        self.pt1 = pt1
        self.pt2 = pt2
        self.normal = Vector.normalize(Vector.sub(pt2, pt1))
        self.thick = 5.0
    
    def get_dist_xy(pt):
        return (abs(Vector.dot_product(Vector.sub(pt, self.pt1), self.normal)))
    
    def get_dist_z():
        # a terme on s'en servira pour avoir un z_buffer
        return (0.0)
    
    def is_inside(pt):
        return (self.get_dist_xy(pt) <= self.thick)

class Object:
    
    def __init__(self):
        self.pts = [] # liste de point
        self.conexion = [] #liste des connexiont
        self.nb_point = len(self.pts) 



###### DATA ######
# {position + axe} camera
# liste des position des point
# liste des liens des point
# ==> liste des segment




###### ALGO ######
# le point en 3d
# changement de repere camera
# projection conique
# tracer une ligne a l'ecran

def get_seg_dist_2d(seg, pt):
	# diff  scalaire normale < largeur 
	print("la on fait de la magie")

def get_color_pix(x, y, size, time):
	cx = (float(x) / size[0])
	cy = (float(y) / size[1])
	value = (m.sin(m.sin(cy * 7.0 + time * 1.01)) + \
		m.cos((cx * 12.0) + 1.0 * time) + \
		m.cos(cx * 37.0 + 1.4 * time) * 0.3 +\
		m.sin(cx * 80.0 + 1.4 * time) * 0.1 +\
		2.4 ) / 4.8
	
	#(m.sin((15.0 * (x + -time)) / size[0]) + m.cos(((7.0 * (y + time))/size[1])) + m.cos(float(x) / size[0] + 1.01 * -time) * 0.1 + 2.5) / 5.0
	return get_col(value)

def print_all_screen(tab_line, size_ecr):
    os.system("clear;")	
    img = ""
    t = tt.clock()
    print("\n")
    for y in range(size_ecr[1]):
        line = ""
        for x in range(size_ecr[0] - 1):
        	line += str(get_color_pix(x, y, size_ecr, t * 10))
        img += line + "\n"
    os.system("echo \""+ img + "\"")



#################################################



def test_palette():
    print("nb lines:", size[0], "nb cols:", size[1])
    for a in range(-2, 10):
        help(a)
        b = a
        print("=====================================>", b)
        val = float((float(b) + 0.0) / 7.0)

def test_termshader():
    ecr = get_ecr_size()
    print("screen:{x:"+str(ecr[0])+", y:"+str(ecr[1])+"}")
    tab_line = [{'x1':0, 'x2':ecr[0] , 'y1':0 , 'y2':ecr[1], 'col':0.9}]
    
    while True :
        print_all_screen(tab_line, get_ecr_size())
        tt.sleep(0.05)
###################################################################################
###################################################################################
###################################################################################
###################################################################################

#lst1 # la premiere liste de point
#lst2 # la deusieme liste de point
#len1 # la premiere taille
#len2 # la deusieme taille

def get_number_face(nb):
    if (nb == 4):
        lst_pt = np.array([[[0.1, 0.1, 0.3],[0.9, 0.1, 0.3],[1.2, 0.5, 0.3],[0.9, 0.9, 0.3],[0.1, 0.1, 0.3]],[[0.1, 0.1, -0.3],[0.9, 0.1, -0.3],[1.2, 0.5, -0.3],[0.9, 0.9, -0.3],[0.1, 0.1, -0.3]]])
    elif (nb == 2):
      lst_nb = np.array([[[0.3, 0.3, 0.3],[0.6, 0.3, 0.3],[1.2, 0.5, 0.3],[0.6, 0.6, 0.3],[0.3, 0.6, 0.3],[0.3, 0.3, 0.3]],[[0.3, 0.3, -0.3],[0.6, 0.3, -0.3],[1.2, 0.5, -0.3],[0.6, 0.6, -0.3],[0.3, 0.6, -0.3],[0.3, 0.3, -0.3]]])
    return (lst_pt)


def get_rotation_matrice(rot):
    rot_all = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    c = np.cos(rot.data[0])
    s = np.sin(rot.data[0])
    rotx = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    c = np.cos(rot.data[1])
    s = np.sin(rot.data[1])
    roty = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    c = np.cos(rot.data[2])
    s = np.sin(rot.data[2])
    rotz = np.array([[0., c, -s], [0., s, c], [0., 0., 1.]])
    rot_all = rot_all.dot(rotx)
    rot_all = rot_all.dot(roty)
    rot_all = rot_all.dot(rotz)
    return (rot_all)

def landmark_change(obj_seg, nb_seg, center, rot):
    for i in range(nb_seg):
        tmp = (obj_seg[0][i].sub(center))
        new_seg[0][i] = tmp.mult(rot)

        tmp = (obj_seg[1][i].sub(center))
        new_seg[1][i] = tmp.mult(rot)
    return (new_seg)


def landmark_change_proj(size, obj_seg, nb_seg, center, rot):
    for i in range(nb_seg):
        tmp = np.add(obj_seg[0][i], center * -1)
        rot_mat = get_rotation_matrice(rot)
        print("rot:", rot_mat)
        tmp = rot_mat.dot(tmp)
        print("tmp", tmp)
        tt.sleep(10000)
        print("tmp.data[0][0]", tmp.data[0][0])
        print("\ttmp.data[0][2]", tmp.data[0][2],)
        print("\tsize.data[0][0]", size.data[0][0])
        tmp.data[0][0] = (tmp.data[0][0] / tmp.data[0][2]) * size.data[0][0]
        tmp.data[0][1] = tmp.data[0][1] / tmp.data[0][2]  * size.data[0][1]
        new_seg[0][i] = tmp

        tmp = np.sub(obj_seg[1][i], center)
        tmp = rot.dot(tmp)
        tmp.data[1][0] = (tmp.data[1][0] / tmp.data[1][2]) * size.data[1][0]
        tmp.data[1][1] = tmp.data[1][1] / tmp.data[1][2]  * size.data[1][1]
        new_seg[1][i] = tmp
    return (new_seg)

def is_in_line(size, pos_ecr, pt_beg, pt_end, width):
    # il faut la projection du point dans le repere de la ligne
    # du coup on peut determiner la normal et donc la distance
    print("yolo\n")

# on retourne eventuelement un nombre entre [0., 1.]
#    ou directement le charactere
def is_in_lst(size, lst, nb_pt, pos_ecr):
    for i in range(nb_pt):
        id_prev = ((i - 1 + nb_pt) % nb_pt)
        id_next = ((i + 1) % nb_pt)
        # lst.data[0][i] -> lst.data[0][id_prev]
        # lst.data[0][i] -> lst.data[0][id_next]
        # lst.data[1][i] -> lst.data[1][id_prev]
        # lst.data[1][i] -> lst.data[1][id_next]
        # lst.data[0][i] -> lst.data[1][i]



# Il faut encore 


def vector_test_print():
    # 
    vmat = [[1., 2.], [3., 4.]]
    vtest = np.array([42.1337, 1.3, .5])
 #   help(vtest)
    print("coucou\n", get_ecr_size())
    print("x=>", vtest.data[0])
    for j in range(2):
        for i in range(2):
            print("-->[", j, "],[", i, "]<--\n")

def vector_test():
   # obj_2 = get_number_face(2)
    size = get_ecr_size_vec()
    nb = 4
    obj_4 = get_number_face(nb)
    cam_pos = np.array([0., 0.,-5])
    len_4 = 4 if (nb == 4) else 2
    rot = np.array([0., 0., 0.])

    obj_tmp = landmark_change_proj(size, obj_4, len_4, cam_pos, rot)
    img_txt = ""
    # pour toute les pixel
    for y in range(size.data[1]):
        img_txt += "\n"
        for x in range(size.data[0]):
            val = is_in_lst(size, obj_tmp, len_4, cam_pos, rot)
            img_txt += get_col(val)
        os.system(("echo " + img_txt))
    #time.sleep(0.025)

# on va renvoyer une liste des point d'une des face sous forme de liste
# on va la copier et la deplacer selon la normale de la face.
# pour chaque point serra relier a 3 autre point: ses deux voisin et son equivalent de l'autre face

if __name__ == "__main__":
#    test_termshader()
    vector_test()


#	word = ""
#	help("")
#	test_line(size)
