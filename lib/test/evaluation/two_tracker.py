import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from lib.test.utils.load_text import load_text

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from rgbt.utils import IoU
from lib.utils.box_ops import giou_loss
import torch
from eval_tracker.small_model import Switcher_v1


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, 
                 name2: str, parameter_name2: str, 
                 dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, checkpoint_path=None, debug=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.name2 = name2
        self.parameter_name2 = parameter_name2
        self.dataset_name = dataset_name
        self.display_name = display_name
        self.checkpoint_path = checkpoint_path
        self.switcher = Switcher_v1()
        self.switcher.load_state_dict(torch.load("./small_model_ep0.pth.tar"))

        if debug:
            # 调取baseline的跟踪结果
            self.baseline_resfile = {
                # "rgbt234":"/home/zhaojiacong/all_result/tracking_result/attn_fusion/cross_attn_fusion_ep030/rgbt234/",
                "rgbt234":"/home/zhaojiacong/all_result/tracking_result/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300_ep084/rgbt234/",
                "lashertestingset":"/home/zhaojiacong/all_result/tracking_result/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300_ep084/lashertestingset/"
            }
            for k in list(self.baseline_resfile.keys()):
                if k in dataset_name.lower():
                    self.baseline_resfile = self.baseline_resfile[k]
                    break
            self.baseline_rect = dict()
            # for seq_name in os.listdir(self.baseline_resfile):
            #     res_path = os.path.join(self.baseline_resfile, seq_name)
            #     seq_name = seq_name.split('.')[0].split('_')[-1]
            #     self.baseline_rect[seq_name] = load_text(str(res_path), delimiter=['', '\t', ','], dtype=np.float64)

        self.params = self.get_parameters()
        self.params2 = self.get_parameters2()
        self.run_id = run_id
        # self.run_id = run_id

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}_{}/{}_{:03d}'.format(env.results_path, self.name, self.name2, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None
            
        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name2))
        if os.path.isfile(tracker_module_abspath):
            tracker_module2 = importlib.import_module('lib.test.tracker.{}'.format(self.name2))
            self.tracker_class2 = tracker_module2.get_tracker_class()
        else:
            self.tracker_class2 = None

    def create_tracker(self, params, params2):
        tracker = self.tracker_class(params, self.dataset_name)
        tracker2 = self.tracker_class2(params2, self.dataset_name)
        return tracker, tracker2

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """

        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_
        self.params2.debug = debug_

        # Get init information
        init_info = seq.init_info()

        trackers = self.create_tracker(params, self.params2)

        output = self._track_sequence(trackers, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': [],
                  'score_map': [],
                  'score_map_2': [],
                  'choice': []}
        if tracker[0].params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []
            output['all_scoremaps'] = []
            output['all_scoremaps_2'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        # image = self._read_image(seq.frames[0])
        image_v = self._read_image(seq.frames_v[0])
        image_i = self._read_image(seq.frames_i[0])

        start_time = time.time()
        #out = tracker.initialize(image, init_info)
        out = tracker[0].initialize(image_v, image_i, init_info) # Initialize network
        out2 = tracker[1].initialize(image_v, image_i, init_info) # Initialize network

        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker[0].params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)
        frame_num =0 
        for frame_path_v, frame_path_i in zip(seq.frames_v[1:], seq.frames_i[1:]):
        #for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            # image = self._read_image(frame_path)

            frame_num += 1
            image_v = self._read_image(frame_path_v) # load each frame image
            image_i = self._read_image(frame_path_i)
            
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output
            try:
                info['baseline_rect'] = self.baseline_rect[seq.name][frame_num]
            except:
                info['baseline_rect'] = [0.,0.,0.,0.]

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker[0].track(image_v, image_i, info)
            out2 = tracker[1].track(image_v, image_i, info)
            out['score_map_2'] = out2['score_map']
            # 真值引导
            # gt = torch.tensor(seq.ground_truth_rect[frame_num][None])
            # gt[:, 2]+=gt[:, 0]; gt[:, 3]+=gt[:, 1]
            # p1 = torch.tensor(out['target_bbox'])[None]
            # p1[:, 2]+=p1[:, 0]; p1[:, 3]+=p1[:, 1]
            # p2 = torch.tensor(out2['target_bbox'])[None]
            # p2[:, 2]+=p2[:, 0]; p2[:, 3]+=p2[:, 1]
            # if giou_loss(gt, p1)[0]<= giou_loss(gt, p2)[0]:
            #     prev_output = OrderedDict(out)
            #     tracker[1].state = out['target_bbox']
            #     out['choice'] = 0
            # else:
            #     prev_output = OrderedDict(out2)
            #     out['target_bbox'] = out2['target_bbox']
            #     tracker[0].state = out2['target_bbox']
            #     out['choice'] = 1
            # Score引导
            if out['score_map'].max() >= out['score_map_2'].max():
            # if self.switcher(torch.cat([out['scoe_map'].reshape(1,1,1,16,16), out['scoe_map_2'].reshape(1,1,1,16,16)], dim=1)).item()<=0:
                prev_output = OrderedDict(out)
                tracker[1].state = out['target_bbox']
                out['choice'] = 0
            else:
                prev_output = OrderedDict(out2)
                out['target_bbox'] = out2['target_bbox']
                tracker[0].state = out2['target_bbox']
                out['choice'] = 1
            _store_outputs(out, {'time': time.time() - start_time})
            
        for key in ['target_bbox', 'all_boxes', 'all_scores', 'all_scoremaps', 'all_scoremaps_2']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name, self.checkpoint_path)
        return params

    def get_parameters2(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name2))
        params = param_module.parameters(self.parameter_name2, None)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



