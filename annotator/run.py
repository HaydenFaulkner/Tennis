"""
Used to annotate temporal events of different classes with particular attributes within a video

Input is a video file and a path to write out the save file
"""

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import json
import numpy as np
import os.path
from PIL import Image, ImageTk
import time
from tkinter import Tk, Label, Frame, GROOVE, Text, FLAT, END, Button, RIGHT, CURRENT
from tkinter.font import Font, BOLD

flags.DEFINE_string('video_file', 'data/videos/V006.mp4',
                    'Path to the video file to annotate.')
flags.DEFINE_string('type_file', 'data/annotations/classes.txt',
                    'Path to the classes type .txt file.')
flags.DEFINE_string('save_path', 'data/annotations',
                    'Directory to save the .json output.')


class Annotator:
    def __init__(self, video_file, type_file, save_path):
        self.in_file = os.path.normpath(video_file)  # make the paths OS (Windows) compatible
        self.type_file = os.path.normpath(type_file)  # make the paths OS (Windows) compatible
        self.out_file = os.path.normpath(save_path)  # make the paths OS (Windows) compatible

        self.autosave = 1  # set autosave flag
        self.fps = 30  # set fps

        # load the types .txt file
        if os.path.exists(self.type_file):
            with open(self.type_file) as f:
                types = f.readlines()
                self.types = [item.rstrip().split("\t") for item in types]  # remove the newline character and split
        else:
            logging.error("No type file found at: {}\nPlease check path and try again.".format(self.type_file))

        # load the video
        if not os.path.isfile(self.in_file):
            logging.error("Video file doesn't exist: {}".format(self.in_file))
        else:
            self.cap = cv2.VideoCapture(self.in_file)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames < 1:
                logging.error("Video file can't be loaded: {}\n"
                              "This may be related to OpenCV and FFMPEG".format(self.in_file))
            else:
                logging.info("Loaded video file successfully: {}".format(self.in_file))

        # let's build the GUI
        self.root = Tk()
        my_font = Font(root=self.root, family="Arial", size=9)
        # self.root.config(font=my_font)
        self.root.title('Video Sequence Annotator')
        geom = "1281x847"
        self.root.geometry(geom)
        self.root.configure(background="#505050")
        self.root.bind("<Key>", self.key)
        self.root.bind('<Left>', self.leftKey)
        self.root.bind('<Right>', self.rightKey)
        self.root.bind('<Up>', self.upKey)
        self.root.bind('<Down>', self.downKey)
        win_h = self.root.winfo_height()
        win_w = self.root.winfo_width()

        self.image_label = Label(master=self.root, font=my_font)
        self.image_label.pack()

        self.timeline_label = Label(master=self.root, font=my_font)
        self.timeline_label.bind("<ButtonPress-1>", self.mouse_down)
        self.timeline_label.bind("<ButtonRelease-1>", self.mouse_up)
        self.timeline_label.bind("<B1-Motion>", self.mouse_move)
        self.timeline_label.bind("<MouseWheel>", self._on_mousewheel)
        self.timeline_label.pack()

        self.current_frame = 0
        self.play_pause_flag = 1
        self.clicked_flag = 0
        self.click_event = None
        self.mouse_x = 0
        self.selected_flag = 0
        self.selected_class = self.types[0][0]
        self.selected_class = ''
        self.selected_index = -1
        self.selected_index_name = ''
        self.selected_index_start = -1
        self.selected_index_end = -1
        self.started_seq_flag = 0
        self.started_seq_name = ''
        self.step_back_flag = 0
        self.step_for_flag = 0
        self.dynamic_labels = {}
        self.dynamic_label_titles = {}
        self.dynamic_label_buts = {}
        self.start_crop = 0
        self.start_time = time.clock()
        self.zoom = 120  # the larger the more zoomed in the timeline
        self.speed = 5

        _, vid_id = os.path.split(self.in_file)
        vid_id = '.'.join(vid_id.split('.')[:-1])

        # load the save json it it already exists
        save_path = os.path.join(self.out_file, vid_id + ".json")
        if os.path.isfile(save_path):
            logging.info("File {} exists. Loading it.".format(save_path))
            with open(save_path, 'r') as f:
                self.database = json.load(f)
        else:
            logging.info("File {} does not exist. Will make new database.".format(save_path))
            self.database = {'video': self.in_file, 'classes': {'USE': [], 'SPLITS': []}}

            if len(self.types) > 0:
                for the_class in self.types:
                    self.database['classes'][the_class[0]] = []

        self.stats_label = Frame(master=self.root)
        self.stats_label.place(relx=0.6, rely=0.01, relheight=0.8, width=250)
        self.stats_label.configure(relief=GROOVE)
        self.stats_label.configure(borderwidth="2")
        self.stats_label.configure(relief=GROOVE)
        self.stats_label.configure(background="#505050")
        self.stats_label.configure(width=275)
        self.stats_label.pack()

        line_label = Frame(master=self.stats_label)
        line_label.place(x=15, y=213, height=1, width=315)
        line_label.configure(background="#999")

        self.tools_label = Frame(master=self.root)
        self.tools_label.place(relx=0.6, rely=0.01, relheight=0.8, width=250)
        self.tools_label.configure(relief=GROOVE)
        self.tools_label.configure(borderwidth="2")
        self.tools_label.configure(relief=GROOVE)
        self.tools_label.configure(background="#505050")
        self.tools_label.pack()

        stats_title = Label(self.stats_label, font=my_font)
        stats_title.place(x=5, y=0, height=30, width=340)
        stats_title.configure(activebackground="#555")
        stats_title.configure(activeforeground="white")
        stats_title.configure(background="#505050")
        stats_title.configure(foreground="#FFF")
        stats_title.configure(text='Default Properties')

        stats_cust_title = Label(self.stats_label, font=my_font)
        stats_cust_title.place(x=5, y=218, height=30, width=340)
        stats_cust_title.configure(activebackground="#555")
        stats_cust_title.configure(activeforeground="white")
        stats_cust_title.configure(background="#505050")
        stats_cust_title.configure(foreground="#FFF")
        stats_cust_title.configure(text='Custom Properties')

        video_name_label_title = Label(self.stats_label, font=my_font)
        video_name_label_title.place(x=5, y=30, height=30, width=50)
        video_name_label_title.configure(activebackground="#555")
        video_name_label_title.configure(activeforeground="white")
        video_name_label_title.configure(background="#505050")
        video_name_label_title.configure(foreground="#FFF")
        video_name_label_title.configure(text='Video:')

        video_name_label = Text(self.stats_label, font=my_font)
        video_name_label.place(x=55, y=33, height=22, width=288)
        video_name_label.configure(background="#666")
        video_name_label.configure(foreground="#FFF")
        video_name_label.configure(relief=FLAT)
        video_name_label.insert(END, self.in_file)
        video_name_label.config(highlightbackground='#505050')
        video_name_label.configure(state='disabled')

        class_name_label_title = Label(self.stats_label, font=my_font)
        class_name_label_title.place(x=5, y=60, height=30, width=50)
        class_name_label_title.configure(activebackground="#555")
        class_name_label_title.configure(activeforeground="white")
        class_name_label_title.configure(background="#505050")
        class_name_label_title.configure(foreground="#FFF")
        class_name_label_title.configure(text='Class:')

        self.class_name_label = Text(self.stats_label, font=my_font)
        self.class_name_label.place(x=55, y=63, height=22, width=288)
        self.class_name_label.configure(background="#666")
        self.class_name_label.configure(foreground="#FFF")
        self.class_name_label.configure(relief=FLAT)
        self.class_name_label.insert(END, self.selected_class)
        self.class_name_label.config(highlightbackground='#505050')
        # class_name_label.configure(state='disabled')

        self.index_name_label_title = Label(self.stats_label, font=my_font)
        self.index_name_label_title.place(x=5, y=90, height=30, width=50)
        self.index_name_label_title.configure(activebackground="#555")
        self.index_name_label_title.configure(activeforeground="white")
        self.index_name_label_title.configure(background="#505050")
        self.index_name_label_title.configure(foreground="#FFF")
        self.index_name_label_title.configure(text='Name:')

        self.index_name_label = Text(self.stats_label, font=my_font)
        self.index_name_label.place(x=55, y=93, height=22, width=288)
        self.index_name_label.configure(background="#666")
        self.index_name_label.configure(foreground="#FFF")
        self.index_name_label.configure(relief=FLAT)
        self.index_name_label.insert(END, self.selected_index_name)
        self.index_name_label.config(highlightbackground='#505050')

        self.index_start_label_title = Label(self.stats_label, font=my_font)
        self.index_start_label_title.place(x=5, y=120, height=30, width=50)
        self.index_start_label_title.configure(activebackground="#555")
        self.index_start_label_title.configure(activeforeground="white")
        self.index_start_label_title.configure(background="#505050")
        self.index_start_label_title.configure(foreground="#FFF")
        self.index_start_label_title.configure(text='Start:')

        self.index_start_label = Text(self.stats_label, font=my_font)
        self.index_start_label.place(x=55, y=123, height=22, width=288)
        self.index_start_label.configure(background="#666")
        self.index_start_label.configure(foreground="#FFF")
        self.index_start_label.configure(relief=FLAT)
        self.index_start_label.insert(END, self.selected_index_name)
        self.index_start_label.config(highlightbackground='#505050')

        self.index_end_label_title = Label(self.stats_label, font=my_font)
        self.index_end_label_title.place(x=5, y=150, height=30, width=50)
        self.index_end_label_title.configure(activebackground="#555")
        self.index_end_label_title.configure(activeforeground="white")
        self.index_end_label_title.configure(background="#505050")
        self.index_end_label_title.configure(foreground="#FFF")
        self.index_end_label_title.configure(text='End:')

        self.index_end_label = Text(self.stats_label, font=my_font)
        self.index_end_label.place(x=55, y=153, height=22, width=288)
        self.index_end_label.configure(background="#666")
        self.index_end_label.configure(foreground="#FFF")
        self.index_end_label.configure(relief=FLAT)
        self.index_end_label.insert(END, self.selected_index_name)
        self.index_end_label.config(highlightbackground='#505050')

        self.index_desc_label_title = Label(self.stats_label, font=my_font)
        self.index_desc_label_title.place(x=5, y=180, height=30, width=50)
        self.index_desc_label_title.configure(activebackground="#555")
        self.index_desc_label_title.configure(activeforeground="white")
        self.index_desc_label_title.configure(background="#505050")
        self.index_desc_label_title.configure(foreground="#FFF")
        self.index_desc_label_title.configure(text='Desc:')

        self.index_desc_label = Text(self.stats_label, font=my_font)
        self.index_desc_label.place(x=55, y=183, height=22, width=288)
        self.index_desc_label.configure(background="#666")
        self.index_desc_label.configure(foreground="#FFF")
        self.index_desc_label.configure(relief=FLAT)
        self.index_desc_label.insert(END, self.selected_index_name)
        self.index_desc_label.config(highlightbackground='#505050')

        self.prev_frame_but = Button(self.tools_label, command=self.prev_frame, font=my_font)
        self.prev_frame_but.configure(activebackground="#d9d9d9")
        self.prev_frame_but.configure(background="#505050")
        self.prev_frame_but.configure(foreground="#FFF")
        self.prev_frame_but.configure(relief=FLAT)
        self.prev_frame_but.configure(text='<')

        self.next_frame_but = Button(self.tools_label, command=self.next_frame, font=my_font)
        self.next_frame_but.configure(activebackground="#d9d9d9")
        self.next_frame_but.configure(background="#505050")
        self.next_frame_but.configure(foreground="#FFF")
        self.next_frame_but.configure(relief=FLAT)
        self.next_frame_but.configure(text='>')

        self.back_but = Button(self.tools_label, command=self.step_back, font=my_font)
        self.back_but.configure(activebackground="#d9d9d9")
        self.back_but.configure(background="#505050")
        self.back_but.configure(foreground="#FFF")
        self.back_but.configure(relief=FLAT)
        self.back_but.configure(text='|<')

        self.skip_but = Button(self.tools_label, command=self.step_forward, font=my_font)
        self.skip_but.configure(activebackground="#d9d9d9")
        self.skip_but.configure(background="#505050")
        self.skip_but.configure(foreground="#FFF")
        self.skip_but.configure(relief=FLAT)
        self.skip_but.configure(text='>|')

        self.play_pause = Button(self.tools_label, command=self.play, font=my_font)
        self.play_pause.configure(activebackground="#d9d9d9")
        self.play_pause.configure(background="#505050")
        self.play_pause.configure(foreground="#FFF")
        self.play_pause.configure(relief=FLAT)
        self.play_pause.configure(text='PAUSE')

        self.time_label = Label(self.tools_label, font=my_font)
        self.time_label.configure(background="#505050")
        self.time_label.configure(foreground="#FFF")
        self.time_label.configure(text='''Time / Total''')
        self.time_label.pack()

        self.del_class_but = Button(self.tools_label, command=self.del_class, font=my_font)
        self.del_class_but.configure(activebackground="#d9d9d9")
        self.del_class_but.configure(background="#505050")
        self.del_class_but.configure(foreground="#FFF")
        self.del_class_but.configure(relief=FLAT)
        self.del_class_but.configure(text='DELETE CLASS')

        self.add_class_but = Button(self.tools_label, command=self.add_class, font=my_font)
        self.add_class_but.configure(activebackground="#d9d9d9")
        self.add_class_but.configure(background="#505050")
        self.add_class_but.configure(foreground="#FFF")
        self.add_class_but.configure(relief=FLAT)
        self.add_class_but.configure(text='NEW CLASS')

        self.del_seq_but = Button(self.tools_label, command=self.del_seq, font=my_font)
        self.del_seq_but.configure(activebackground="#d9d9d9")
        self.del_seq_but.configure(background="#505050")
        self.del_seq_but.configure(foreground="#FFF")
        self.del_seq_but.configure(relief=FLAT)
        self.del_seq_but.configure(text='DELETE')

        self.start_seq_but = Button(self.tools_label, command=self.start_seq, font=my_font)
        self.start_seq_but.configure(activebackground="#d9d9d9")
        self.start_seq_but.configure(background="#505050")
        self.start_seq_but.configure(foreground="#FFF")
        self.start_seq_but.configure(relief=FLAT)
        self.start_seq_but.configure(text='START')

        self.end_seq_but = Button(self.tools_label, command=self.end_seq, font=my_font)
        self.end_seq_but.configure(activebackground="#d9d9d9")
        self.end_seq_but.configure(background="#505050")
        self.end_seq_but.configure(foreground="#FFF")
        self.end_seq_but.configure(relief=FLAT)
        self.end_seq_but.configure(text='END')

        self.update_data_but = Button(self.stats_label, command=self.update_data, font=my_font)
        self.update_data_but.configure(activebackground="#d9d9d9")
        self.update_data_but.configure(background="#505050")
        self.update_data_but.configure(foreground="#FFF")
        self.update_data_but.configure(relief=FLAT)
        self.update_data_but.configure(text='UPDATE')

        self.refresh_all = True

        # set the colour values
        self.colours = [(4, 174, 248), (16, 217, 4), (240, 191, 16), (217, 48, 14), (133, 4, 248),
                        (16, 2, 248), (2, 217, 165), (184, 240, 14), (217, 131, 13), (248, 2, 109)]
        self.highlights = [(151, 217, 248), (135, 217, 132), (240, 223, 158), (217, 156, 143), (204, 151, 248),
                           (150, 150, 248), (131, 217, 191), (225, 240, 157), (217, 183, 142), (248, 150, 203)]

        # setup the update callback
        self.root.after(self.speed, func=lambda: self.update_all())
        self.root.mainloop()

        for the_class in self.database['classes']:
            num_del = 0
            for i in range(0, len(self.database['classes'][the_class])):
                start_frame = self.database['classes'][the_class][i-num_del]['start']
                end_frame = self.database['classes'][the_class][i-num_del]['end']
                if end_frame - start_frame < 5:
                    num_del += 1
                    print('Deleted ' + self.database['classes'][the_class][i]['name'])
                    del self.database['classes'][the_class][i]

        self.save()

    def save(self):
        _, vid_id = os.path.split(self.in_file)

        with open(os.path.join(self.out_file, vid_id+".json"), 'w') as f:
            json.dump(self.database, f)

    def quit_(self):
        self.save()
        self.root.destroy()

    def update_timeline(self):
        # Setup Variables
        currentframe = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        try:
            win_h = self.root.winfo_height()
            win_w = self.root.winfo_width()
        except Exception:
            self.save()
            return
        d_h = int((win_h-100)-(.6*win_h))
        d_w = int(max((win_w-20), (self.zoom*((self.total_frames/25)/60))))

        each_line_h = int((d_h-50)/max((len(self.database['classes'])-2)*1.0, 1.0))

        # Do logic first then Draw
        # a sequence has been started keep updating the end point (refresh_some should be T)
        if self.started_seq_flag == 1:
            self.selected_index_end = self.current_frame
            self.database['classes'][self.selected_class][self.selected_index]['end'] = currentframe

            # Refresh some .. only the selected sequence
            self.refresh_some = True
            outer_index = 0
            inner_index = 0
            for the_class in self.database['classes']:

                if the_class == 'USE':
                    continue
                elif the_class == 'SPLITS':
                    continue
                else:
                    pos_n = int(30+outer_index*each_line_h + 5)
                    pos_s = int(30+(outer_index+1)*each_line_h - 5)

                    for i in range(0, len(self.database['classes'][the_class])):
                        start_frame = self.database['classes'][the_class][i]['start']
                        end_frame = self.database['classes'][the_class][i]['end']
                        pos_w = int(d_w*start_frame/self.total_frames)
                        pos_e = int(d_w*end_frame/self.total_frames)

                        if (self.selected_flag == 1) & (self.selected_class == the_class) & (self.selected_index == i):
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = self.highlights[outer_index % 10]
                        else:
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = self.colours[outer_index % 10]

                        inner_index += 1
                    outer_index += 1

        # a click has occurred
        if self.clicked_flag == 1:
            self.update_data()
            found_flag = 0
            cx = self.click_event.x
            cy = self.click_event.y

            # if in timeline skipper
            if cy > (d_h-10):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(((self.start_crop + cx) / (1.0 * d_w)) * self.total_frames) - 1)

                self.read_single = 1
                self.past = currentframe  # could just use curframe var
                self.currentframe = int(((self.start_crop+cx)/(1.0*d_w))*self.total_frames)-1
                # will need to redraw where line was!!! HOW? USE PAST
                self.refresh_all = True  # change later

            # if scrollbar
            elif cy < 11:
                # if on actual scrollbar
                if (cx > int(self.start_crop/d_w)) and (cx < int(win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))):
                    bar_width = int((win_w-20)*((win_w-20)/(1.0*d_w)))
                    self.start_crop = ((1.0*min(max(0, self.mouse_x), int(win_w-20)-bar_width)/int(win_w-20))*d_w)
                    # Refresh all
                    self.refresh_all = True
            else:
                # if clicked in an area other than scroll or skip bars then stop sequence iff it's been started
                if self.started_seq_flag == 1:
                    self.end_seq()
                # find the selected class
                outer_index = 0
                if (cy > (d_h-20)) and (cy < (d_h-10)):
                    self.selected_class = 'USE'
                elif (cy > 10) and (cy < 30):
                    self.selected_class = 'SPLITS'
                else:

                    for the_class in sorted(self.database['classes']):
                        if the_class == 'USE' or the_class == 'SPLITS':
                            continue
                        else:
                            if (cy > (30 + outer_index*each_line_h + 5)) and \
                                    (cy < (30+(outer_index+1)*each_line_h - 5)):
                                self.selected_class = the_class
                                break
                            outer_index += 1

                # find the selected sequence
                inner_index = 0

                adj_cx = self.click_event.x+self.start_crop
                for i in range(0, len(self.database['classes'][self.selected_class])):
                    start_frame = self.database['classes'][self.selected_class][i]['start']
                    end_frame = self.database['classes'][self.selected_class][i]['end']

                    if (adj_cx > (d_w*start_frame/self.total_frames)) and (adj_cx < (d_w*end_frame/self.total_frames)):
                        self.selected_index = i
                        self.selected_index_name = self.database['classes'][self.selected_class][i]['name']
                        self.selected_index_start = self.database['classes'][self.selected_class][i]['start']
                        self.selected_index_end = self.database['classes'][self.selected_class][i]['end']
                        found_flag = 1
                        break

                    inner_index += 1

            if (found_flag == 0) & (self.started_seq_flag == 0):
                if (cy < (d_h-10)) & (cy > 10):
                    self.selected_flag = 0
                    self.selected_index = ''
                    self.selected_index_name = ''
                    self.selected_index_start = ''
                    self.selected_index_end = ''
            else:
                self.selected_flag = 1

            self.refresh_all = True
            self.update_labels()

        # COMPLETELY REDRAW
        if self.refresh_all:  # repaint entire timeline
            self.display = np.zeros((d_h, d_w, 3))
            self.display = self.display.astype('uint8')
            self.display[:] = 50
            self.display[10:30] = 90

            # paint selected class backgound
            outer_index = 0
            for the_class in sorted(self.database['classes']):
                if the_class == 'USE' or the_class == 'SPLITS':
                    continue
                elif self.selected_class == the_class:
                    self.display[(30 + outer_index*each_line_h):(30+(outer_index+1)*each_line_h)] = 75

                self.display[30+outer_index*each_line_h-1:30+outer_index*each_line_h+1] = 100
                self.display[d_h-21:d_h-19] = 200

                outer_index += 1

            # paint all sequences
            outer_index = 0
            inner_index = 0
            for the_class in sorted(self.database['classes']):  # alphabetical order to class displays

                if the_class == 'USE':
                    pos_n = d_h-20
                    pos_s = d_h-10
                    self.display[pos_n:pos_s, :, :] = (130, 0, 0)
                    for i in range(0, len(self.database['classes'][the_class])):
                        start_frame = self.database['classes'][the_class][i]['start']
                        end_frame = self.database['classes'][the_class][i]['end']
                        pos_w = int(d_w*start_frame/self.total_frames)
                        pos_e = int(d_w*end_frame/self.total_frames)

                        if (self.selected_flag == 1) & (self.selected_class == the_class) & (self.selected_index == i):
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = (0, 230, 0)
                        else:
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = (0, 130, 0)

                        self.display[10:d_h-20, pos_w-1:pos_w, :] = (250, 0, 0)
                        self.display[10:d_h-20, pos_e:pos_e+1, :] = (250, 0, 0)
                        inner_index += 1

                elif the_class == 'SPLITS':
                    pos_n = 12
                    pos_s = 28
                    for i in range(0, len(self.database['classes'][the_class])):
                        start_frame = self.database['classes'][the_class][i]['start']
                        end_frame = self.database['classes'][the_class][i]['end']
                        pos_w = int(d_w*start_frame/self.total_frames)
                        pos_e = int(d_w*end_frame/self.total_frames)

                        if (self.selected_flag == 1) & (self.selected_class == the_class) & (self.selected_index == i):
                            if self.database['classes'][self.selected_class][self.selected_index]['custom']['Type']\
                                    == 'Train':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (0, 190, 0)
                            elif self.database['classes'][self.selected_class][self.selected_index]['custom']['Type']\
                                    == 'Val':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (190, 190, 0)
                            elif self.database['classes'][self.selected_class][self.selected_index]['custom']['Type']\
                                    == 'Test':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (190, 0, 0)
                            else:
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (40, 40, 40)
                        else:
                            if self.database['classes'][the_class][i]['custom']['Type'] == 'Train':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (0, 130, 0)
                            elif self.database['classes'][the_class][i]['custom']['Type'] == 'Val':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (130, 130, 0)
                            elif self.database['classes'][the_class][i]['custom']['Type'] == 'Test':
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (130, 0, 0)
                            else:
                                self.display[pos_n:pos_s, pos_w:pos_e, :] = (0, 0, 0)

                        inner_index += 1
                else:
                    pos_n = int(30+outer_index*each_line_h + 5)
                    pos_s = int(30+(outer_index+1)*each_line_h - 5)

                    for i in range(0, len(self.database['classes'][the_class])):
                        start_frame = self.database['classes'][the_class][i]['start']
                        end_frame = self.database['classes'][the_class][i]['end']
                        pos_w = int(d_w*start_frame/self.total_frames)
                        pos_e = int(d_w*end_frame/self.total_frames)

                        if (self.selected_flag == 1) & (self.selected_class == the_class) & (self.selected_index == i):
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = self.highlights[outer_index % 10]
                        else:
                            self.display[pos_n:pos_s, pos_w:pos_e, :] = self.colours[outer_index % 10]

                        inner_index += 1
                    outer_index += 1

            # these 4 fors are a bottleneck
            for i in range(0, d_w, int(self.zoom/2.0)):
                for j in range(10, d_h, 4):
                    self.display[j:j+2, i:i+1] = 105

            for i in range(0, d_w, int(self.zoom*2.5)):
                self.display[10:, i:i+1] = 150

            for i in range(0, d_w, int(self.zoom*30)):
                self.display[10:, i-1:i+1] = 160

            self.display[d_h-10:d_h-9, :] = 200

            self.display[d_h-10:d_h, :int(d_w*self.current_frame/self.total_frames), :] = 200  # prog bar

        display_cropped = self.display[:, int(self.start_crop):int((self.start_crop+int(win_w-20))), :]
        display_cropped[0:10, :] = 50
        display_cropped[9:10, :] = 200
        display_cropped[0:10, int((win_w-20)*(self.start_crop/(d_w*1.0))):
                              int((win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))), :] = 200  # scroll bar
        display_cropped[2:8, int((win_w-20)*(self.start_crop/(d_w*1.0)) +
                                 (int(win_w-20)*((self.start_crop+int(win_w-20)) / (1.0*d_w)) -
                                  int(win_w-20)*(self.start_crop/(d_w*1.0)))/2.0), :] = 50  # scroll bar
        display_cropped[2:8, int((win_w-20)*(self.start_crop/(d_w*1.0)) - 2 +
                                 (int(win_w-20)*((self.start_crop+int(win_w-20)) / (1.0*d_w)) -
                                  int(win_w-20)*(self.start_crop/(d_w*1.0)))/2.0), :] = 50  # scroll bar
        display_cropped[2:8, int((win_w-20)*(self.start_crop/(d_w*1.0)) + 2 +
                                 (int(win_w-20)*((self.start_crop+int(win_w-20)) / (1.0*d_w)) -
                                  int(win_w-20)*(self.start_crop/(d_w*1.0)))/2.0), :] = 50  # scroll bar
        for i in range(0, 5):
            display_cropped[0:5-i, int((win_w-20)*(self.start_crop/(d_w*1.0))):
                                   int((win_w-20)*(self.start_crop/(d_w*1.0)) + i), :] = 50
            display_cropped[4+i:9, int((win_w-20)*(self.start_crop/(d_w*1.0))):
                                   int((win_w-20)*(self.start_crop/(d_w*1.0)) + i), :] = 50
            display_cropped[0:5-i, int((win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))-i):
                                   int((win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))), :] = 50
            display_cropped[4+i:9, int((win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))-i):
                                   int((win_w-20)*((self.start_crop+int(win_w-20))/(1.0*d_w))), :] = 50

        a = Image.fromarray(display_cropped)

        b = ImageTk.PhotoImage(image=a)
        self.timeline_label.configure(image=b)
        self.timeline_label.place(x=10, y=20+int(.6*win_h))

        self.timeline_label._image_cache = b  # avoid garbage collection
        # clicked_flag = 0

        self.refresh_all = False
        self.root.update()

    def update_labels(self):

        self.class_name_label.configure(state='normal')
        self.class_name_label.delete(1.0, END)
        self.class_name_label.insert(CURRENT, self.selected_class)

        self.index_name_label.configure(state='normal')
        self.index_name_label.delete(1.0, END)
        self.index_start_label.configure(state='normal')
        self.index_start_label.delete(1.0, END)
        self.index_end_label.configure(state='normal')
        self.index_end_label.delete(1.0, END)
        self.index_desc_label.configure(state='normal')
        self.index_desc_label.delete(1.0, END)

        for key in self.dynamic_labels.keys():
            self.dynamic_labels[key].destroy()

        for key in self.dynamic_label_titles.keys():
            self.dynamic_label_titles[key].destroy()

        for key in self.dynamic_label_buts.keys():
            self.dynamic_label_buts[key].destroy()

        self.dynamic_labels = {}
        self.dynamic_label_titles = {}
        self.dynamic_label_buts = {}

        if self.selected_flag > 0:

            self.index_name_label.insert(CURRENT, self.database['classes'][self.selected_class][self.selected_index]['name'])
            self.index_start_label.insert(CURRENT, '%02d:%02d:%02d.%02d' %
                                          (int(((self.selected_index_start/self.fps)/60)/60),
                                           int(((self.selected_index_start/self.fps)/60) % 60),
                                           int((self.selected_index_start/self.fps) % 60),
                                           int((self.selected_index_start % self.fps))))
            self.index_end_label.insert(CURRENT, '%02d:%02d:%02d.%02d' %
                                        (int(((self.selected_index_end/self.fps)/60)/60),
                                         int(((self.selected_index_end/self.fps)/60) % 60),
                                         int((self.selected_index_end/self.fps) % 60),
                                         int((self.selected_index_end % self.fps))))
            self.index_desc_label.insert(CURRENT, self.database['classes'][self.selected_class][self.selected_index]['desc'])

            self.cust = self.database['classes'][self.selected_class][self.selected_index]['custom']

            i = 0
            for key in self.cust.keys():
                self.dynamic_label_titles[key] = Label(self.stats_label)
                self.dynamic_label_titles[key].place(x=5, y=(248+(i*30)), height=30, width=50)
                self.dynamic_label_titles[key].configure(activebackground="#555")
                self.dynamic_label_titles[key].configure(activeforeground="white")
                self.dynamic_label_titles[key].configure(background="#505050")
                self.dynamic_label_titles[key].configure(foreground="#FFF")
                self.dynamic_label_titles[key].configure(text=(key+':'))

                self.dynamic_labels[key] = Text(self.stats_label)
                self.dynamic_labels[key].place(x=55, y=(251+(i*30)), height=22, width=260)
                self.dynamic_labels[key].configure(background="#666")
                self.dynamic_labels[key].configure(foreground="#FFF")
                self.dynamic_labels[key].configure(relief=FLAT)
                self.dynamic_labels[key].config(highlightbackground='#505050')
                self.dynamic_labels[key].configure(state='normal')
                self.dynamic_labels[key].delete(1.0, END)
                self.dynamic_labels[key].insert(CURRENT, self.cust[key])

                self.dynamic_label_buts[key] = Button(self.stats_label, command=lambda key=key: self.del_custom(key))
                self.dynamic_label_buts[key].place(x=320, y=(251+(i*30)), height=22, width=22)
                self.dynamic_label_buts[key].configure(activebackground="#d9d9d9")
                self.dynamic_label_buts[key].configure(background="#505050")
                self.dynamic_label_buts[key].configure(foreground="#FFF")
                self.dynamic_label_buts[key].configure(relief=FLAT)
                self.dynamic_label_buts[key].configure(text='-')

                i += 1

            self.dynamic_labels['_ADD_'] = Text(self.stats_label)
            self.dynamic_labels['_ADD_'].place(x=55, y=(251+(i*30)), height=22, width=260)
            self.dynamic_labels['_ADD_'].configure(background="#666")
            self.dynamic_labels['_ADD_'].configure(foreground="#FFF")
            self.dynamic_labels['_ADD_'].configure(relief=FLAT)
            self.dynamic_labels['_ADD_'].insert(END, 'New_cust')
            self.dynamic_labels['_ADD_'].config(highlightbackground='#505050')

            self.dynamic_label_buts['_ADD_'] = Button(self.stats_label, command=lambda: self.add_custom(
                self.dynamic_labels['_ADD_'].get(1.0, END)))
            self.dynamic_label_buts['_ADD_'].place(x=320, y=(251+(i*30)), height=22, width=22)
            self.dynamic_label_buts['_ADD_'].configure(activebackground="#d9d9d9")
            self.dynamic_label_buts['_ADD_'].configure(background="#505050")
            self.dynamic_label_buts['_ADD_'].configure(foreground="#FFF")
            self.dynamic_label_buts['_ADD_'].configure(relief=FLAT)
            self.dynamic_label_buts['_ADD_'].configure(text='+')

            self.stats_label.pack()

        self.save()

    def update_data(self):

        if (self.selected_class is not None) & (self.selected_class != ''):
            self.database['classes'][str(self.class_name_label.get(1.0, END)).replace('\n', '')] = \
                self.database['classes'].pop(self.selected_class, None)
            self.selected_class = str(self.class_name_label.get(1.0, END)).replace('\n', '')

            if (self.selected_index is not None) & (self.selected_index != ''):
                self.database['classes'][self.selected_class][self.selected_index]['name'] = \
                    str(self.index_name_label.get(1.0, END)).replace('\n', '')
                self.database['classes'][self.selected_class][self.selected_index]['desc'] = \
                    str(self.index_desc_label.get(1.0, END)).replace('\n', '')

                self.cust = self.database['classes'][self.selected_class][self.selected_index]['custom']
                for key in self.cust:
                    self.cust[key] = str(self.dynamic_labels[key].get(1.0, END)).replace('\n', '')

        self.save()

    def del_custom(self, key):
        self.update_data()
        if (self.selected_class is not None) & (self.selected_class != '') & (self.selected_class != 'SPLITS'):
            if (self.selected_index is not None) & (self.selected_index != ''):
                self.database['classes'][self.selected_class][self.selected_index]['custom'].pop(key, None)

        self.update_labels()

    def add_custom(self, key):
        self.update_data()
        if (self.selected_class is not None) & (self.selected_class != '') & (self.selected_class != 'SPLITS'):
            if (self.selected_index is not None) & (self.selected_index != ''):
                self.database['classes'][self.selected_class][self.selected_index]['custom'][str(key).replace('\n', '')] = ''

        self.update_labels()

    def add_class(self):
        self.refresh_all = True
        self.database['classes']['New_Class'] = []

    def del_class(self):
        self.refresh_all = True
        if (self.selected_class is not None) & (self.selected_class != ''):  # & (self.selected_class != 'SPLITS'):
            print(self.selected_class)
            self.database['classes'].pop(self.selected_class, None)
            self.selected_flag = 0
            self.update_labels()

    def del_seq(self):
        self.refresh_all = True
        if (self.selected_class is not None) & (self.selected_class != '') &\
                (self.selected_index is not None) & (self.selected_index != ''):
            del self.database['classes'][self.selected_class][self.selected_index]
            self.selected_flag = 0
            self.selected_index = None
            self.update_labels()

    def start_seq(self):
        self.refresh_all = True
        if (self.selected_class is not None) & (self.selected_class != ''):

            if self.selected_flag == 1:
                if self.database['classes'][self.selected_class][self.selected_index]['end'] > self.current_frame:
                    self.database['classes'][self.selected_class][self.selected_index]['start'] = self.current_frame
                else:
                    taken_names = []
                    for i in range(0, len(self.database['classes'][self.selected_class])):
                        taken_names.append(self.database['classes'][self.selected_class][i]['name'])

                    new_name = 'error'
                    for i in range(1, 1000000):  # shouldn't have more than this many events in a class per video...
                        if "%04d" % i not in taken_names:
                            new_name = "%04d" % i
                            break

                    if self.selected_class == 'SPLITS':
                        self.database['classes'][self.selected_class].append({'name': new_name,
                                                                              'start': int(self.current_frame),
                                                                              'end': int(self.current_frame)+1,
                                                                              'desc': '',
                                                                              'custom': {'Type': 'Train'}})
                    else:
                        self.database['classes'][self.selected_class].append({'name': new_name,
                                                                              'start': int(self.current_frame),
                                                                              'end': int(self.current_frame)+1,
                                                                              'desc': '',
                                                                              'custom': {}})
                        if len(self.types) > 0:
                            for the_class in self.types:  # check all classes
                                if len(the_class) > 1:  # if has attributes
                                    if self.selected_class == the_class[0]:  # and if is selected
                                        attr = the_class[1].split(",")
                                        for the_attr in attr:
                                            self.database['classes'][self.selected_class][len(self.database['classes'][self.selected_class])-1]['custom'][str(the_attr).replace('\n', '')] = ''

                                            self.update_labels()

                    self.selected_flag = 1
                    self.started_seq_flag = 1
                    self.selected_index_name = 'New_name'
                    self.selected_index = len(self.database['classes'][self.selected_class])-1

                    self.selected_index_start = int(self.current_frame)
                    self.selected_index_end = int(self.current_frame)+1
            else:
                taken_names = []
                for i in range(0, len(self.database['classes'][self.selected_class])):
                    taken_names.append(self.database['classes'][self.selected_class][i]['name'])

                new_name = 'error'
                for i in range(1, 1000000):  # shouldnt have more than this many events in a class per video...
                    if "%04d" % i not in taken_names:
                        new_name = "%04d" % i
                        break

                if self.selected_class == 'SPLITS':
                    self.database['classes'][self.selected_class].append({'name': new_name,
                                                                          'start': int(self.current_frame),
                                                                          'end': int(self.current_frame)+1,
                                                                          'desc': '',
                                                                          'custom': {'Type': 'Train'}})
                else:
                    self.database['classes'][self.selected_class].append({'name': new_name,
                                                                          'start': int(self.current_frame),
                                                                          'end': int(self.current_frame)+1,
                                                                          'desc': '',
                                                                          'custom': {}})
                    if len(self.types) > 0:
                        for the_class in self.types:  # check all classes
                            if len(the_class) > 1:  # if has attributes
                                if self.selected_class == the_class[0]:  # and if is selected
                                    attr = the_class[1].split(",")
                                    for the_attr in attr:
                                        self.database['classes'][self.selected_class][len(self.database['classes'][self.selected_class])-1]['custom'][str(the_attr).replace('\n', '')] = ''

                                        self.update_labels()

                self.selected_flag = 1
                self.started_seq_flag = 1
                self.selected_index_name = 'New_name'
                self.selected_index = len(self.database['classes'][self.selected_class])-1

                self.selected_index_start = int(self.current_frame)
                self.selected_index_end = int(self.current_frame)+1

            self.update_labels()

    def end_seq(self):
        self.refresh_all = True
        if (self.selected_class is not None) & (self.selected_class != '') & (self.selected_flag == 1):
            if self.current_frame > self.database['classes'][self.selected_class][self.selected_index]['start']:
                self.database['classes'][self.selected_class][self.selected_index]['end'] = int(self.current_frame)
                self.started_seq_flag = 0
                self.update_labels()

    def update_image(self):
        if self.play_pause_flag > 0:
            (readsuccessful, f) = self.cap.read()
            if (readsuccessful == 'False') | (f is None):
                print('No Video file')
                f = self.past_f

                self.play_pause_flag = 0
                self.play_pause.configure(text='PLAY')
            else:
                self.past_f = f

                self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        else:

            if (self.next_frame_flag > 0) | (self.read_single > 0):
                (readsuccessful, f) = self.cap.read()
                if (readsuccessful == 'False') | (f is None):
                    f = self.past_f
                else:
                    self.past_f = f
                    self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            elif self.prev_frame_flag > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 2))

                (readsuccessful, f) = self.cap.read()
                if (readsuccessful == 'False') | (f is None):
                    f = self.past_f
                else:
                    self.past_f = f
                    self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            elif self.step_back_flag > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 26))

                (readsuccessful, f) = self.cap.read()
                if (readsuccessful == 'False') | (f is None):
                    f = self.past_f
                else:
                    self.past_f = f
                    self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            elif self.step_for_flag > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,
                             min(self.total_frames, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 24))

                (readsuccessful, f) = self.cap.read()
                if (readsuccessful == 'False') | (f is None):
                    f = self.past_f
                else:
                    self.past_f = f
                    self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                f = self.past_f

        self.read_single = 0
        self.next_frame_flag = 0
        self.prev_frame_flag = 0
        self.step_back_flag = 0
        self.step_for_flag = 0

        orig_i_h, orig_i_w, orig_i_c = f.shape
        win_h = self.root.winfo_height()
        win_w = self.root.winfo_width()
        r_i_w = 20
        r_i_h = 15

        if win_w > 1:
            r_i_w = min(int(win_w-380), int(.6*win_h*(1.0*orig_i_w/orig_i_h)))
            r_i_h = min(int(.6*win_h), int((win_w-380)*(1.0*orig_i_h/orig_i_w)))
            f = cv2.resize(f, (r_i_w, r_i_h))
        gray_im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        a = Image.fromarray(gray_im)

        b = ImageTk.PhotoImage(image=a)
        self.image_label.configure(image=b)
        # keep in middle
        self.image_label.place(x=max(10, int(((win_w-380)-r_i_w)/2.0)), y=max(10, int(((win_h*.6)-r_i_h)/2.0)))

        self.image_label._image_cache = b  # avoid garbage collection
        self.root.update()

    def mouse_down(self, event):
        self.clicked_flag = 1
        self.click_event = event

    def mouse_up(self, event):
        self.clicked_flag = 0
        self.click_event = event

    def mouse_move(self, event):
        self.mouse_x = event.x

    def _on_mousewheel(self, event):
        print(event.delta)

    def leftKey(self):
        self.speed = min(100,int(self.speed*2.0))

    def rightKey(self):
        self.speed = max(1,int(self.speed/2.0))

    def upKey(self):
        self.zoom = int(self.zoom*1.5)  # the larger the more zoomed in the timeline
        self.refresh_all = True

    def downKey(self):
        self.zoom = int(self.zoom/1.5)  # the larger the more zoomed in the timeline
        self.refresh_all = True

    def key(self, event):
        # print event.char
        # print event.keysym
        # if event.char == ' ':

        if event.keysym == 'F1':
            self.start_seq()
        elif event.keysym == 'F2':
            self.end_seq()
        elif event.keysym == 'F5':
            if self.play_pause_flag > 0:
                self.play_pause_flag = 0
                self.play_pause.configure(text='PLAY')
            else:
                self.play_pause_flag = 1
                self.play_pause.configure(text='PAUSE')

    def update_stats(self):
        try:
            win_h = self.root.winfo_height()
            win_w = self.root.winfo_width()
        except Exception:
            self.save()
            return
        self.stats_label.place(x=win_w-360, y=10, height=int(.6*win_h), width=350)

    def update_tools(self):
        try:
            win_h = self.root.winfo_height()
            win_w = self.root.winfo_width()
        except Exception:
            self.save()
            return
        self.tools_label.place(x=10, y=win_h-70, height=60, width=win_w-20)

    def play(self):
        if self.play_pause_flag > 0:
            self.play_pause_flag = 0
            self.play_pause.configure(text='PLAY')
        else:
            self.play_pause_flag = 1
            self.play_pause.configure(text='PAUSE')

    def next_frame(self):
        if self.next_frame_flag > 0:
            self.next_frame_flag = 0
        else:
            self.next_frame_flag = 1

    def prev_frame(self):
        if self.prev_frame_flag > 0:
            self.prev_frame_flag = 0
        else:
            self.prev_frame_flag = 1

    def step_back(self):
        if self.step_back_flag > 0:
            self.step_back_flag = 0
        else:
            self.step_back_flag = 1

    def step_forward(self):
        if self.step_for_flag > 0:
            self.step_for_flag = 0
        else:
            self.step_for_flag = 1

    def update_all(self):

        # self.save()
        # print 'Starting update'
        # start_inner = time.clock()
        # latest = start_inner
        # try:
        self.update_image()
        # print 'Update Image'
        # taken = time.clock()-latest
        # latest = time.clock()
        # print taken
        self.update_timeline()
        self.update_stats()
        self.update_tools()
        try:
            win_h = self.root.winfo_height()
            win_w = self.root.winfo_width()
        except Exception:
            self.save()
            return

        # except Exception as e:
        #
        #     print("Error: update_all() ; %s" % e)
        #     return

        self.time_label.configure(text='%02d:%02d:%02d.%02d | %02d:%02d:%02d.%02d' %
                                       (int(((self.current_frame/self.fps)/60)/60),
                                        int(((self.current_frame/self.fps)/60) % 60),
                                        int((self.current_frame/self.fps) % 60),
                                        int((self.current_frame % self.fps)),
                                        int(((self.total_frames/self.fps)/60)/60),
                                        int(((self.total_frames/self.fps)/60) % 60),
                                        int((self.total_frames/self.fps) % 60),
                                        int((self.total_frames % self.fps))))
        self.time_label.place(x=win_w-210, y=15, height=30, width=180)
        self.time_label.configure(justify=RIGHT)

        self.play_pause.place(x=int((win_w/2.0)-30), y=12, height=30, width=60)

        self.next_frame_but.place(x=int((win_w/2.0)+35), y=12, height=30, width=40)

        self.prev_frame_but.place(x=int((win_w/2.0)-75), y=12, height=30, width=40)

        self.skip_but.place(x=int((win_w/2.0)+80), y=12, height=30, width=40)

        self.back_but.place(x=int((win_w/2.0)-120), y=12, height=30, width=40)

        self.add_class_but.place(x=10, y=12, height=30, width=100)

        self.del_class_but.place(x=115, y=12, height=30, width=100)

        self.start_seq_but.place(x=220, y=12, height=30, width=70)

        self.end_seq_but.place(x=295, y=12, height=30, width=70)

        self.del_seq_but.place(x=370, y=12, height=30, width=70)

        self.update_data_but.place(x=125, y=self.stats_label.winfo_height()-40, height=30, width=100)

        if (time.clock()-self.start_time) > 60:
            self.start_time = time.clock()
            _, vid_id = os.path.split(self.in_file)

            with open(os.path.join(self.out_file, "autosaves", vid_id+'_A'+str(self.autosave)+'.json'), 'w') as f:
                json.dump(self.database, f)

            if self.autosave == 1:
                self.autosave = 2
            else:
                self.autosave = 1

        try:
            self.root.after(self.speed, func=lambda: self.update_all())
        except Exception as e:
            print("Error: root.after() ; %s" % e)


def main(_argv):
    Annotator(FLAGS.video_file, FLAGS.type_file, FLAGS.save_path)


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
