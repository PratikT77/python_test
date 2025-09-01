"""
Loads Data Set API methods
"""
################################################################################
import os
import time

import numpy as np
from fp_util.CONHUL import CONHUL_point, Convex_Hull
from fp_util.Path_Func import Path_Func

from .els_data import ELS_data
from .geom_basic_data import GEOM_Basic_data
from .geom_os_data import GEOM_OS_data
from .lre1d_data import LRE1D_data
from .lre2d_data import LRE2D_data
from .lrtc_data import LRTC_data, info_name_map
from .lrtc_sparse_data import LRTC_sparse_data

################################################################################

axis_text = ["x", "y", "z"]
axis_index = {"x": 0, "y": 1, "z": 2}


################################################################################
class ATLAS_Data_Set(object):
    """
    ATLAS_Data_Set object for ATLAS format files
    """

    # =============================================================================
    def __init__(self, **args):
        """
        Constructor of ATLAS_Data_Set object for ATLAS format files (geom_basic, geom_os, els, lrtc, lre1d, lre2d, hdf5, env.csv, etc...)

        It can be initiallised empty:
            atlas_obj = ATLAS_Data_Set()

        Or it can be initiallised with content data:

        - from files (lrtc, hdf5, els, geom_os, geom_basic, clgm, lre1d, lre2d, env1d, env2d):
            atlas_obj = ATLAS_Data_Set(lrtc=lrtc_file, els=els_file, geom_os=geom_os_file, geom_basic=geom_basic_file)
            atlas_obj = ATLAS_Data_Set(hdf5=hdf5_file, els=els_file)

        - from class objects (lrtc_obj, hdf5_obj, els_obj, clgm_obj, lre1d_obj, lre2d_obj, env1d_obj, env2d_obj):
            atlas_obj = ATLAS_Data_Set(lrtc_obj=<lrtc_object>, els_obj=<els_object>)
            atlas_obj = ATLAS_Data_Set(hdf5_obj=<hdf5_object>)

        Note: if geom_os is not given using a lrtc_file or lrtc_obj, it must be set as argument: skip_geometry=True
        Note: if hdf5_file or hdf5_obj contains an ELS, there is no need to be given as argument

        - with a data dictionary containing the following structure:
            atlas_obj = ATLAS_Data_Set(data=ATLAS_dict, els=els)

            Where ATLAS_dict can be defined as:
                ATLAS_dict = {"content": {"lrtc":    <lrtc_object> or DATASET_dict,
                                          "aero":    <lrtc_object> or DATASET_dict,
                                          "inertia": <lrtc_object> or DATASET_dict,
                              "info": {"Aircraft":         "A350",       # Not mandatory
                                       "Project":          "PA43L0C1",   # Not mandatory
                                       "DB_Name":          "my_MERGE",   # Not mandatory
                                       "Date":             "30/04/1974"  # Not mandatory
                                       "creation_program": "pyGASO",     # Not mandatory
                                       "LimitUltimate":    "LIMIT",      # Mandatory
                                       },
                              }
            and DATASET_dict can be defined as:
                DATASET_dict = {
                                "LRC_values": [[2.0, 0.0], [0.0, 2.0]],
                                "IQ_names": ["FU.0010.16", "FU.0010.24"],
                                "LRC_names": ["PA43L0T1.MVPER111.CVPE001001", "PA43L0T1.MVPER111.CVPE__LRC3"],
                                "MassCase_names": ["FYYY", "FZZZ"]
                                }

        """
        # Imported inside this function to avoid circular dependencies
        from .HDF5_Data_Set import HDF5_Data_Set, get_common_LRC_list

        self.name = ""
        self.file_name = ""
        self.info = {"Units": "SI"}

        self.has_model_data = False
        self.has_LRC_values = False
        self.has_env1d_data = False
        self.has_env2d_data = False
        self.env2d_dict = None
        self.env1d_IQ_rank_lists = None
        self.calc_env_on_the_fly = False

        self.IQ_names = []

        self.LRC_names = []
        self.SRC_names = None
        self.MassCase_names = []
        self.LRC_index = {}
        self.LRC_name_file_index = {}
        self.LRC_matrix_type = ""
        self.LRC_mirror_def_data = None

        self.LRC_file_list = []
        self.LRC_file_index = {}

        self.LRC_mod_mode = ""
        self.new_LRC_names = {}

        # ---------------------------------------------------------------------

        self.data_source_files = {}

        self.geom_basic_obj = None
        self.geom_os_obj = None
        self.clgm_obj = None
        self.els_obj = None

        self.is_model_modified = False

        self.lrtc_obj = None
        self.lre1d_obj = None
        self.lre2d_obj = None

        self.is_data_modified = False

        # ---------------------------------------------------------------------

        self.verbose = args.get("verbose", False)

        # ---------------------------------------------------------------------

        if not self.load_model_data(**args):
            return

        # ---------------------------------------------------------------------

        def add_LRC_name(local_LRC_name, local_mass_case, local_lrtc_file):
            if LRC_name in self.LRC_index:
                LRC_index = self.LRC_index[local_LRC_name]
            else:
                LRC_index = self.add_LRC_name(local_LRC_name, local_mass_case, local_lrtc_file)

            return LRC_index

        # ---------------------------------------------------------------------

        if "lrtc" in args:
            lrtc_file = args["lrtc"]
            if os.path.exists(lrtc_file):
                ds_obj = self.open_lrtc_file(lrtc_file)
                if ds_obj is not None:

                    self.lrtc_obj = ds_obj
                    self.lrtc_header = getattr(ds_obj, "IQ_names", [])

                    self.name = lrtc_file
                    self.file_name = lrtc_file
                    self.has_LRC_values = True
                    self.LRC_matrix_type = "full"
                    self.data_source_files["lrtc"] = lrtc_file

            elif lrtc_file.lower().strip().startswith("model"):
                self.name = lrtc_file

        if "lrtc_obj" in args:
            lrtc_obj = args['lrtc_obj']

            self.lrtc_obj = lrtc_obj
            self.lrtc_header = getattr(lrtc_obj, "IQ_names", [])

            self.name = lrtc_obj.file_name
            self.file_name = lrtc_obj.file_name
            self.has_LRC_values = True
            self.LRC_matrix_type = "full"
            self.data_source_files["lrtc"] = lrtc_obj.file_name

        if "hdf5" in args:
            hdf5_file = args["hdf5"]
            if os.path.exists(hdf5_file):
                hdf5_obj = HDF5_Data_Set()
                hdf5_obj.hdf5_file_obj = hdf5_obj.open_file(hdf5_file)
                extra_ds = hdf5_obj.get_extra_datasets()

                if hdf5_obj is not None:
                    self.name = hdf5_file
                    self.file_name = hdf5_file
                    self.data_source_files["hdf5"] = hdf5_file

                    if not self.geom_basic_obj:
                        self.geom_basic_obj = hdf5_obj.geom_basic_obj
                        self.data_source_files["geom_basic"] = hdf5_obj.data_source_files.get("geom_basic", "")
                    if not self.geom_os_obj:
                        self.geom_os_obj = hdf5_obj.geom_os_obj
                        self.data_source_files["geom_os"] = hdf5_obj.data_source_files.get("geom_os", "")
                    if not self.els_obj:
                        self.els_obj = hdf5_obj.els_obj
                        self.data_source_files["els"] = hdf5_obj.data_source_files.get("els", "")

                    for ds in extra_ds:
                        setattr(self, ds + "_obj", getattr(hdf5_obj, ds + "_obj", None))
                        setattr(self, ds + "_header", getattr(hdf5_obj, ds + "_header", None))

                    self.has_LRC_values = True
                    self.LRC_matrix_type = "full"

            elif hdf5_file.lower().strip().startswith("model"):
                self.name = hdf5_file

        if "hdf5_obj" in args:
            hdf5_obj = args['hdf5_obj']

            for extra_ds in hdf5_obj.get_extra_datasets():
                setattr(self, extra_ds + "_obj", getattr(hdf5_obj, extra_ds + "_obj", None))

            self.name = hdf5_obj.file_name
            self.file_name = hdf5_obj.file_name
            self.has_LRC_values = True
            self.LRC_matrix_type = "full"
            self.data_source_files["hdf5"] = hdf5_obj.file_name

        if "atlas_obj" in args:
            atlas_obj = args['atlas_obj']
            # Store the content of each element  in atlas_obj in self (the atlas object we are creating)
            for elem, value in atlas_obj.__dict__.items():
                setattr(self, elem, value)

        if "data" in args:
            # ATLAS_Data_Set content is given with a dictionary
            # This dictionary could contain ATLAS objects or another dictionary with the necessary information:
            # (LRC_Names, MassCase_names, IQ_names, LRC_values)

            data = args["data"]

            if "info" not in data and "LimitUltimate" not in data["info"]:
                print("ATLAS_Data_Set definition Error: at least data['info']['LimitUltimate'] must be defined")
                return

            if 'content' in data:
                # Check if all datasets in "content" has the same LRCs.
                # If not, create a common list of LRC to store in LOADCASES and CASES
                LRC_common_list, SRC_common_list, MassCase_common_list, are_common_LRCs = get_common_LRC_list(data)

                for ds in data['content'].keys():
                    if are_common_LRCs:
                        # Write LRTC content if dataset is LRTC_type
                        if isinstance(data['content'][ds], (LRTC_data, HDF5_Data_Set, ATLAS_Data_Set)):
                            setattr(self, ds + "_obj", data['content'][ds])
                            setattr(self, ds + "_header", data['content'][ds].IQ_names)
                        elif isinstance(data['content'][ds], dict):
                            # Use LRTC_data as object to store the data
                            ds_obj = LRTC_data()
                            # IQ and LRC names of the LRC value matrix
                            ds_obj.IQ_names = data['content'][ds]['IQ_names'][:]
                            ds_obj.LRC_names = data['content'][ds]['LRC_names'][:]
                            ds_obj.MassCase_names = data['content'][ds]['MassCase_names'][:]
                            ds_obj.LRC_data = data['content'][ds]['LRC_values']
                            for ind_IQ, IQ_name in enumerate(ds_obj.IQ_names):
                                ds_obj.IQ_index[IQ_name] = ind_IQ
                            for ind_LRC, LRC_name in enumerate(ds_obj.LRC_names):
                                ds_obj.LRC_index[LRC_name] = ind_LRC

                            ds_obj.set_info(data["info"])

                            setattr(self, ds + "_obj", ds_obj)
                            setattr(self, ds + "_header", ds_obj.IQ_names)

                        else:
                            print("ATLAS_Data_Set definition Error: data[content] is not an ATLAS object or a Dictionary")
                            return

                    else:  # are not common LRCs
                        if isinstance(data['content'][ds], (LRTC_data, HDF5_Data_Set, ATLAS_Data_Set)):
                            ds_obj = data['content'][ds]
                            ds_LRC_names = ds_obj.LRC_names
                            if not isinstance(ds_obj.LRC_data, np.ndarray) and not ds_obj.LRC_data:
                                ds_obj.load_LRC_data()
                            ds_LRC_values = ds_obj.LRC_data
                        elif isinstance(data['content'][ds], dict):
                            ds_obj = LRTC_data()
                            ds_obj.IQ_names = data['content'][ds]['IQ_names'][:]
                            ds_LRC_names = data['content'][ds]['LRC_names'][:]
                            ds_LRC_values = data['content'][ds]['LRC_values']
                        else:
                            print("ATLAS_Data_Set definition Error: data[content] is not an ATLAS object or a Dictionary")
                            return

                        ds_obj.LRC_names = LRC_common_list
                        ds_obj.SRC_names = SRC_common_list
                        ds_obj.MassCase_names = MassCase_common_list

                        # Create the LRC_values array for the LRC_common_list
                        LRC_values_common = []
                        for LRC_name in LRC_common_list:
                            if LRC_name in ds_LRC_names:
                                LRC_row = ds_LRC_values[ds_LRC_names.index(LRC_name)]
                            else:
                                LRC_row = [None] * len(ds_obj.IQ_names)
                            LRC_values_common.append(LRC_row)

                        ds_obj.LRC_data = LRC_values_common

                        ds_obj.set_info(data["info"])

                        setattr(self, f"{ds}_obj", ds_obj)
                        setattr(self, f"{ds}_header", ds_obj.IQ_names)

                self.set_info_data(data["info"])

                self.LRC_names = []
                self.MassCase_names = []
                self.LRC_index = {}

                for ind_LRC, LRC_name in enumerate(LRC_common_list):
                    add_LRC_name(LRC_name, MassCase_common_list[ind_LRC], "from_data")

                if "file_name" in args:
                    self.name = args["file_name"]
                    self.file_name = args["file_name"]
                    self.data_source_files["hdf5"] = args["file_name"]

            if getattr(self, "lrtc_obj") and self.lrtc_obj is not None:
                self.has_LRC_values = True
                self.LRC_matrix_type = "full"

        if "lre1d" in args:
            lre1d_file = args["lre1d"]

            if os.path.exists(lre1d_file):
                lre1d_obj = self.load_lre1d_file(lre1d_file)

                if lre1d_obj is not None:
                    self.lre1d_obj = lre1d_obj
                    if not self.name:
                        self.name = lre1d_file
                        self.file_name = lre1d_file
                    self.has_env1d_data = True
                    self.data_source_files["lre1d"] = lre1d_file

        if "lre1d_obj" in args:
            lre1d_obj = args['lre1d_obj']

            if lre1d_obj is not None:
                self.lre1d_obj = lre1d_obj
                if not self.name:
                    self.name = getattr(lre1d_obj, "file_name", "")
                    self.file_name = getattr(lre1d_obj, "file_name", "")
                self.has_env1d_data = True
                self.data_source_files["lre1d"] = getattr(lre1d_obj, "file_name", "")

        if "lre2d" in args:
            lre2d_file = args["lre2d"]

            if os.path.exists(lre2d_file):
                lre2d_obj = self.load_lre2d_file(lre2d_file)

                if lre2d_obj is not None:
                    if not self.name:
                        self.name = lre2d_file
                        self.file_name = lre2d_file
                    self.lre2d_obj = lre2d_obj
                    self.has_env2d_data = True
                    self.data_source_files["lre2d"] = lre2d_file

        if "lre2d_obj" in args:
            lre2d_obj = args["lre2d_obj"]

            if lre2d_obj is not None:
                if not self.name:
                    self.name = getattr(lre2d_obj, "file_name", "")
                    self.file_name = getattr(lre2d_obj, "file_name", "")
                self.lre2d_obj = lre2d_obj
                self.has_env2d_data = True
                self.data_source_files["lre2d"] = getattr(lre2d_obj, "file_name", "")

        if "env1d" in args:
            # TODO: define reading for env1d.csv files
            print('env1d.csv file loading not implemented yet...')
            pass

        if "env1d_obj" in args:
            # TODO: define reading for env1d.csv files
            print('env1d_obj use not implemented yet...')
            pass

        if "env2d" in args:
            # TODO: define reading for env2d.csv files
            print('env2d.csv file loading not implemented yet...')
            pass

        if "env2d_obj" in args:
            # TODO: define reading for env2d.csv files
            print('env2d_obj use not implemented yet...')
            pass

        # ---------------------------------------------------------------------

        if self.lrtc_obj:
            # After reading the inputs provided for the ATLAS_Data_Set inicialisation,
            #  check if there is an ELS object, if not, lrtc_obj (comming from a LRTC, HDF5 or data) can not be managed...
            if not self.els_obj:
                print("ATLAS_Data_Set definition Error: for lrtc content, an els or els_obj must be defined")
                return

            # Add all info keys from the lrtc_obj to the ATLAS_Data_Set object (self)
            self.set_info_data(self.lrtc_obj.get_info())

            lrtc_IQ_names = self.lrtc_obj.get_IQ_names()

            for IQ_name in lrtc_IQ_names:
                self.add_IQ_name(IQ_name)

            LRC_names = self.lrtc_obj.get_LRC_names()
            MassCase_names = self.lrtc_obj.get_MassCase_names()
            lrtc_file_name = self.lrtc_obj.file_name

            self.LRC_names = []
            self.MassCase_names = []
            self.LRC_index = {}
            self.LRC_file_list = []
            self.LRC_file_index = {}
            self.LRC_name_file_index = {}

            for ind_LRC, LRC_name in enumerate(LRC_names):
                add_LRC_name(LRC_name, MassCase_names[ind_LRC], lrtc_file_name)

            self.calc_env_on_the_fly = True

            # check and load croref file
            croref_file = str(lrtc_file_name) + ".croref"
            if os.path.exists(croref_file):
                if self.load_croref_data(croref_file):
                    self.data_source_files["croref"] = croref_file

        # ---------------------------------------------------------------------

        if self.lre1d_obj:
            lre1d_info = self.lre1d_obj.get_info()

            if not self.lrtc_obj:
                self.info["Aircraft"] = lre1d_info["Aircraft"]
                self.info["Project"] = lre1d_info["Project"]
                self.info["DB_name"] = lre1d_info["DB_name"]
                self.info["Date"] = lre1d_info["Date"]
                self.info["LimUlt"] = lre1d_info["LimitUltimate"]

            lre1d_IQ_names = self.lre1d_obj.get_IQ_names()

            for IQ_name in lre1d_IQ_names:
                if not self.add_IQ_name(IQ_name):
                    continue

                rank_desc_list = self.lre1d_obj.get_IQ_rank_desc_list(IQ_name)

                for rank_desc in rank_desc_list:
                    (
                        IQ_value,
                        LRC_name,
                        mass_case,
                        lrtc_file,
                    ) = self.lre1d_obj.get_IQ_rank_data(IQ_name, rank_desc)

                    add_LRC_name(LRC_name, mass_case, lrtc_file)

        # ---------------------------------------------------------------------

        if self.lre2d_obj:
            lre2d_info = self.lre2d_obj.get_info()

            if not self.lrtc_obj and not self.lre1d_obj:
                self.info["Aircraft"] = lre2d_info["Aircraft"]
                self.info["Project"] = lre2d_info["Project"]
                self.info["DB_name"] = lre2d_info["DB_name"]
                self.info["Date"] = lre2d_info["Date"]
                self.info["LimUlt"] = lre2d_info["LimitUltimate"]

            lre2d_env_dict = self.lre2d_obj.get_env_dict()

            for IQ_name1 in lre2d_env_dict.keys():
                if not self.add_IQ_name(IQ_name1):
                    continue

                for IQ_name2 in lre2d_env_dict[IQ_name1].keys():
                    if not self.add_IQ_name(IQ_name2):
                        continue

                    env_data = self.lre2d_obj.get_env_data(IQ_name1, IQ_name2)

                    for i_corner in range(env_data["n_corners"]):
                        LRC_name = env_data["LRC_names"][i_corner]
                        mass_case = env_data["mass_cases"][i_corner]
                        lrtc_file = env_data["lrtc_files"][i_corner]

                        add_LRC_name(LRC_name, mass_case, lrtc_file)

        # ---------------------------------------------------------------------

        if "name" in args:
            self.name = args["name"]

    # =========================================================================
    def load_model_data(self, **args):
        """
        load the model data from geom_basic, geom_os and els
        """
        is_complete_dataset = True if "hdf5" in args or "data" in args or "atlas_obj" in args else False

        if "geom_basic" in args:
            geom_basic_file = args["geom_basic"]
        else:
            file_path = os.path.dirname(os.path.realpath(__file__))
            geom_basic_file = os.path.join(file_path, "Std_Comp_Def_data.ads")

        if geom_basic_file:
            self.geom_basic_obj = self.load_geom_basic_file(geom_basic_file)

            if self.geom_basic_obj:
                self.data_source_files["geom_basic"] = geom_basic_file
            else:
                return False

        if "geom_os" in args:
            geom_os_file = args["geom_os"]
            self.geom_os_obj = self.load_geom_os_file(geom_os_file)

            if self.geom_os_obj:
                self.data_source_files["geom_os"] = geom_os_file
            else:
                return False

        if "clgm" in args:
            # TODO: read the new geometry format from FD&S
            pass

        if "clgm_obj" in args:
            # TODO: read the new geometry format from FD&S
            pass

        if "els" in args:
            els_file = args["els"]
            self.els_obj = self.load_els_file(els_file)

            if self.els_obj:
                self.data_source_files["els"] = els_file
            else:
                return False

        if "els_obj" in args:
            els_obj = args["els_obj"]
            self.els_obj = els_obj

            if self.els_obj:
                self.data_source_files["els"] = els_obj.file_name
            else:
                return False

        # ---------------------------------------------------------------------

        if args.get("skip_geometry", False):
            if not (self.geom_basic_obj and self.els_obj):
                return False
        elif (not (self.geom_basic_obj and self.els_obj and self.geom_os_obj)
              and not (self.geom_basic_obj and self.els_obj and self.clgm_obj)
              and not is_complete_dataset):
            return False

        self.complete_model_set()

        return True

    # =========================================================================
    def load_geom_basic_file(self, file_name):
        """
        load the GEOM_Basic data
        """
        #  check file name
        if not file_name:
            print("load_geom_basic_file: Error - Empty file name")
            return None

        #  check file existance
        if not os.path.exists(file_name):
            print(f"load_geom_basic_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load GEOM_Basic: {file_name}")

        return GEOM_Basic_data(file_name)

    # =========================================================================
    def load_geom_os_file(self, file_name):
        """
        load the GEOM_OS data
        """
        #  check file name
        if not file_name:
            print("load_geom_os_file: Error - Empty file name")
            return None

        #  check file existance
        if not os.path.exists(file_name):
            print(f"load_geom_os_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load GEOM_OS: {file_name}")

        return GEOM_OS_data(file_name)

    # =========================================================================
    def load_els_file(self, file_name):
        """
        load the ELS data
        """
        #  check file name
        if not file_name:
            print("load_els_file: Error - Empty file name")
            return None

        #  check file existance
        if not os.path.exists(file_name):
            print(f"load_els_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load ELS: {file_name}")

        return ELS_data(file_name)

    # =========================================================================
    def open_lrtc_file(self, file_name):
        """
        open a lrtc file
        """
        # check file name
        if not file_name:
            print("open_lrtc_file: Error - Empty file name")
            return None

        # check file existance
        if not os.path.exists(file_name):
            print(f"open_lrtc_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load LRTC: {file_name}")

        return LRTC_data(file_name)

    # =========================================================================
    def load_croref_data(self, file_name):
        """
        load croref file data
        """
        # check file name
        if not file_name:
            return None

        # check file existance
        if not os.path.exists(file_name):
            return None

        if self.verbose:
            print(f"Load croref data: {file_name}")

        # open file
        try:
            file_obj = open(file_name, "r", encoding="latin-1")
        except IOError:
            print(f"load_croref_data: Error - Unable to open file {file_name}")
            return None

        file_lines = file_obj.readlines()

        file_obj.close()

        # read and set the SRC names
        for line in file_lines:
            if not line or len(line) < 71:
                continue
            SRC_name = line[:40]
            LRC_name = line[43:71].ljust(40)
            self.set_SRC_name(LRC_name, SRC_name)

        return True

    # =========================================================================
    def load_lre1d_file(self, file_name):
        """
        open a lre1d file
        """
        # check file name
        if not file_name:
            print("load_lre1d_file: Error - Empty file name")
            return None

        # check file existance
        if not os.path.exists(file_name):
            print(f"load_lre1d_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load LRE1D: {file_name}")

        return LRE1D_data(file_name)

    # =========================================================================
    def load_lre2d_file(self, file_name):
        """
        open a lre2d file
        """
        # check file name
        if not file_name:
            print("load_lre2d_file: Error - Empty file name")
            return None

        # check file existance
        if not os.path.exists(file_name):
            print(f"load_lre2d_file: Error - File not found {file_name}")
            return None

        if self.verbose:
            print(f"Load LRE2D: {file_name}")

        return LRE2D_data(file_name)

    # =========================================================================
    def load_LRC_mirror_def_data(self, mirror_def_type, mirror_def_file):
        """
        load the mirror definition data from a file
        mirror_def_type can be: ELS or DEF
        """
        self.LRC_mirror_def_data = None

        if mirror_def_type == "ELS":
            mirror_def_data = self.load_ELS_mirror_def_data(mirror_def_file)
        else:
            mirror_def_data = self.load_mirror_def_file(mirror_def_file)

        if mirror_def_data is None or len(mirror_def_data) == 0:
            return False

        self.LRC_mirror_def_data = mirror_def_data

        return True

    # =========================================================================
    def load_mirror_def_file(self, mirror_def_file):
        """
        load the mirror definition data from a file
        """
        # check file name
        if not mirror_def_file:
            print("load_mirror_def_data: Error - Empty file name")
            return None

        # check file existance
        if not os.path.exists(mirror_def_file):
            print(f"load_mirror_def_data: Error - File not found {mirror_def_file}")
            return None

        # open file and read content

        try:
            file_obj = open(mirror_def_file, "r", encoding="latin-1")
        except IOError:
            print(f"load_mirror_def_data: Error - Unable to open file: {mirror_def_file}")
            return None

        # read all lines of the file
        file_lines = file_obj.readlines()

        # close the file
        file_obj.close()

        # -----------------------------------------------------------------------

        mirror_def_data = []

        read_def = False

        for ind_line, line in enumerate(file_lines):
            line = line.strip()

            num_line = ind_line + 1

            if line.startswith("[begin_def_mirror]"):
                read_def = True
                continue

            if line.startswith("[end_def_mirror]"):
                break

            if not read_def:
                continue

            # interprete a definition line

            fields = line.split()

            if len(fields) < 3:
                print(f"load_mirror_def_data: [{num_line}]: {line}")
                print("*** invalid definition (number of line entries < 3) ***")

            def1 = fields[0].strip()
            def2 = fields[1].strip()
            factor = float(fields[2])

            if def1.count(".") != 2 or def2.count(".") != 2:
                print(f"load_mirror_def_data: [{num_line}]: {line}")
                print("*** invalid definition (format of line entries) ***")

            (comp_key1, station1, ISO1) = def1.split(".")
            (comp_key2, station2, ISO2) = def2.split(".")

            if (
                station1 == "####" and station2 == "####"
            ):  # definition for all component stations
                comp_ISO_data1 = self.els_obj.get_plot_data2(comp_key1, int(ISO1))
                if comp_ISO_data1 is None:
                    continue

                comp_ISO_data2 = self.els_obj.get_plot_data2(comp_key2, int(ISO2))
                if comp_ISO_data2 is None:
                    continue

                IQ1_names = comp_ISO_data1["IQ_names"]

                for ind_IQ1, IQ1_name in enumerate(IQ1_names):
                    IQ1_station_name = comp_ISO_data1["IQ_stations"][ind_IQ1]
                    (IQ1_comp_key, IQ1_station_id) = IQ1_station_name.split(".")

                    IQ2_station_name = comp_key2 + "." + IQ1_station_id

                    if IQ2_station_name in comp_ISO_data2["IQ_stations"]:
                        ind_station2 = comp_ISO_data2["IQ_stations"].index(
                            IQ2_station_name
                        )
                        IQ2_name = comp_ISO_data2["IQ_names"][ind_station2]

                        if IQ1_name != IQ2_name or factor != 1.0:
                            mirror_def_data.append([IQ1_name, IQ2_name, factor])

            elif (
                station1 == "####"
                and station2 != "####"
                or station1 != "####"
                and station2 == "####"
            ):
                print(f"load_mirror_def_data: [{num_line}]: {line}")
                print("*** invalid definition (mix of IQ_name and all stations) ***")

            else:  # definition for explicit IQ names
                IQ1_name = def1
                IQ2_name = def2

                if IQ1_name != IQ2_name or factor != 1.0:
                    mirror_def_data.append([IQ1_name, IQ2_name, factor])

        # -----------------------------------------------------------------------

        return mirror_def_data

    # =========================================================================
    def load_ELS_mirror_def_data(self, ELS_file=""):
        # check file name
        if not ELS_file:
            mirror_def_els_obj = self.els_obj

        else:
            # check file existance
            if not os.path.exists(ELS_file):
                print(f"load_ELS_mirror_def_data: Error - File not found {ELS_file}")
                return None

            mirror_def_els_obj = ELS_data(ELS_file)

        return mirror_def_els_obj.get_mirror_def_data()

    # =========================================================================
    def get_LRC_mirror_def_data(self):
        return self.LRC_mirror_def_data

    # ===========================================================================
    def add_IQ(self, IQ_attr):
        IQ_def = self.els_obj.add_IQ(IQ_attr)

        self.add_IQ_name(IQ_def.name)

        return IQ_def

    # ===========================================================================
    def add_IQ_name(self, IQ_name):
        if IQ_name not in self.IQ_names:
            if not self.els_obj.has_IQ(IQ_name):
                if self.verbose:
                    print(f"IQ {IQ_name} is missing in the ELS!!")
                return False

            self.IQ_names.append(IQ_name)

        return True

    # ===========================================================================
    def add_LRC_name(self, LRC_name, mass_case, lrtc_file=None):
        if LRC_name in self.LRC_index:
            LRC_index = self.LRC_index[LRC_name]
            if mass_case != "" and self.MassCase_names[LRC_index] != mass_case:
                self.MassCase_names[LRC_index] = mass_case
            return LRC_index

        self.LRC_names.append(LRC_name)
        self.MassCase_names.append(mass_case)

        LRC_index = len(self.LRC_names) - 1
        self.LRC_index[LRC_name] = LRC_index

        if lrtc_file is None:
            lrtc_index = None
        else:
            if lrtc_file in self.LRC_file_index:
                lrtc_index = self.LRC_file_index[lrtc_file]
            else:
                self.LRC_file_list.append(lrtc_file)
                lrtc_index = len(self.LRC_file_list) - 1
                self.LRC_file_index[lrtc_file] = lrtc_index

        self.LRC_name_file_index[LRC_name] = lrtc_index

        return LRC_index

    # ===========================================================================
    def set_SRC_name(self, LRC_name, SRC_name):
        if not LRC_name or not SRC_name:
            return

        if LRC_name in self.LRC_index:
            if self.SRC_names is None:
                self.SRC_names = {}

            self.SRC_names[LRC_name] = SRC_name

    # ===========================================================================
    def rename_LRC(self, LRC_name, new_LRC_name):
        LRC_name = LRC_name.ljust(40)[:40]
        new_LRC_name = new_LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"rename_LRC: invalid LRC name {LRC_name} ({new_LRC_name})")
            return False

        if not new_LRC_name or new_LRC_name == LRC_name:
            print(f"rename_LRC: invalid new LRC name {new_LRC_name} ({LRC_name})")
            return False

        if new_LRC_name in self.LRC_index:
            print(f"rename_LRC: new LRC name already exists {new_LRC_name} ({LRC_name})")
            return False

        ind_LRC = self.LRC_index[LRC_name]

        self.LRC_names[ind_LRC] = new_LRC_name

        del self.LRC_index[LRC_name]

        self.LRC_index[new_LRC_name] = ind_LRC

        lrtc_index = self.LRC_name_file_index[LRC_name]

        self.LRC_name_file_index[new_LRC_name] = lrtc_index

        del self.LRC_name_file_index[LRC_name]

        if self.lrtc_obj is not None:
            self.lrtc_obj.rename_LRC(LRC_name, new_LRC_name)

        if self.lre1d_obj is not None:
            self.lre1d_obj.rename_LRC(LRC_name, new_LRC_name)

        if self.lre2d_obj is not None:
            self.lre2d_obj.rename_LRC(LRC_name, new_LRC_name)

        if self.SRC_names is not None and LRC_name in self.SRC_names:
            SRC_name = self.SRC_names[LRC_name]
            del self.SRC_names[LRC_name]
            self.SRC_names[new_LRC_name] = SRC_name

        return True

    # ===========================================================================
    def delete_LRC(self, LRC_name):
        LRC_name = LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"delete_LRC: invalid LRC name {LRC_name}")
            return False

        self.LRC_names.remove(LRC_name)

        del self.LRC_name_file_index[LRC_name]

        self.remove_env_data()

        if self.lrtc_obj is not None:
            self.lrtc_obj.delete_LRC(LRC_name)

        self.LRC_index = {}

        for ind_LRC, LRC_name in enumerate(self.LRC_names):
            self.LRC_index[LRC_name] = ind_LRC

        if self.SRC_names is not None and LRC_name in self.SRC_names:
            del self.SRC_names[LRC_name]

        return True

    # ===========================================================================
    def get_mirror_LRC_name(self, LRC_name, mod_char="X", return_none_if_exists=True):
        if LRC_name not in self.LRC_index:
            return None

        if LRC_name in self.new_LRC_names:
            return None

        new_LRC_name = self.lrtc_obj.get_mirror_LRC_name(LRC_name, mod_char=mod_char, return_none_if_exists=return_none_if_exists)

        return new_LRC_name

    # ===========================================================================
    def get_mod_LRC_name(self, LRC_name):
        if LRC_name not in self.LRC_index:
            return None

        if LRC_name in self.new_LRC_names:
            return self.new_LRC_names[LRC_name]

        if self.LRC_mod_mode == "modify":
            new_LRC_name = LRC_name

        elif self.LRC_mod_mode == "rename":
            new_LRC_name = self.lrtc_obj.get_mod_LRC_name(LRC_name, "F")
            if new_LRC_name is None:
                return None

            self.rename_LRC(LRC_name, new_LRC_name)
            self.new_LRC_names[LRC_name] = new_LRC_name

        elif self.LRC_mod_mode == "duplicate":
            mass_case = self.lrtc_obj.get_LRC_MassCase(LRC_name)
            new_LRC_name = self.lrtc_obj.get_mod_LRC_name(LRC_name, "F")
            if new_LRC_name is None:
                return None

            self.lrtc_obj.duplicate_LRC(LRC_name, new_LRC_name)
            self.add_LRC_name(new_LRC_name, mass_case)
            self.new_LRC_names[LRC_name] = new_LRC_name

        else:
            new_LRC_name = LRC_name

        return new_LRC_name

    # ===========================================================================
    def complete_model_set(self):
        """
        check and reorganize the model data (els, geom_basic, geom_os) to be as
        consistent as possible + sorting of the stations within the components
        """
        # ---------------------------------------------------------------------
        if "geom_basic_obj" not in dir(self) or "geom_os_obj" not in dir(self) or "els_obj" not in dir(self):
            return

        if self.geom_basic_obj and self.geom_os_obj and self.els_obj:

            self.geom_os_obj.sort_stations(self.geom_basic_obj)

            error_data = self.els_obj.set_geom_model(
                self.geom_basic_obj, self.geom_os_obj
            )

            if error_data is not None:
                print(error_data)

            self.has_model_data = True

    # ===========================================================================
    def exec_mod_cmd_list(self, mod_cmd_list, **kwargs):
        """
        set data set modifications (e.g. station def, IQ_names, ...)
        """
        # ---------------------------------------------------------------------

        cmd_file_path = None

        if "cmd_file_path" in kwargs:
            cmd_file_path = kwargs["cmd_file_path"]

        # ---------------------------------------------------------------------

        self.new_LRC_names = {}

        cmd_list = mod_cmd_list[:]

        while len(cmd_list) > 0:
            cmd_line = cmd_list.pop(0)
            cmd_parts = cmd_line.split(",")
            cmd_param = [s.strip() for s in cmd_parts]
            cmd_key = cmd_param.pop(0)

            for ind in range(len(cmd_param)):
                if cmd_param[ind] == "":
                    cmd_param[ind] = None

            if cmd_key == "set_station_def":
                station_ref = cmd_param[0]
                station_id = cmd_param[1]
                x_coord = cmd_param[2]
                y_coord = cmd_param[3]
                z_coord = cmd_param[4]
                station_desc = cmd_param[5]

                print(f"set_station_def: {station_ref} => "
                      f"{station_id}, ({x_coord},{y_coord},{z_coord}) - {station_desc}")

                station_def = {
                    "Station_ID": station_id,
                    "Coordinates": [x_coord, y_coord, z_coord],
                    "Station_Description": station_desc,
                }

                station_obj = self.geom_os_obj.set_station_def(station_ref, station_def)

                if station_obj is None:
                    print("Error during set_station_def")
                    continue

                self.els_obj.update_from_station_def(station_obj)

                self.is_model_modified = True

            elif cmd_key == "new_station":
                comp_key = cmd_param[0]
                station_id = cmd_param[1]
                x_coord = cmd_param[2]
                y_coord = cmd_param[3]
                z_coord = cmd_param[4]
                station_desc = cmd_param[5]

                print(f"new_station: {comp_key} {station_id}, ({x_coord},{y_coord},{z_coord}) - {station_desc}")

                station_def = {
                    "Station_ID": station_id,
                    "Coordinates": [x_coord, y_coord, z_coord],
                    "Station_Description": station_desc,
                }

                station_obj = self.geom_os_obj.add_station(comp_key, station_def)

                if station_obj is None:
                    print("Error during new_station")
                    continue

                self.is_model_modified = True

            elif cmd_key == "shift_comp_stations":
                station_ref = cmd_param[0]
                dx = cmd_param[1]
                dy = cmd_param[2]
                dz = cmd_param[3]

                print(f"shift_comp_stations: {station_ref} = {dx},{dy},{dz}")

                comp_key = self.geom_os_obj.shift_comp_stations(station_ref, dx, dy, dz)

                self.els_obj.update_from_station_coords(comp_key)

                self.is_model_modified = True

            elif cmd_key == "delete_station":
                station_ref = cmd_param[0]

                print(f"delete_station: {station_ref}")

                if self.is_model_modified:
                    self.reorganize_model_data()

                station_obj = self.geom_os_obj.get_station_by_ref(station_ref)

                if station_obj is None:
                    print(f"Error during delete_station: invalid station ref. {station_ref}")
                    continue

                station_name = station_obj.name

                IQ_names_at_station = self.get_IQ_names_at_station(station_name)
                if IQ_names_at_station is not None and len(IQ_names_at_station) > 0:
                    for IQ_name in IQ_names_at_station:
                        print("delete_IQ: {IQ_name}")
                        if self.delete_IQ(IQ_name):
                            self.is_model_modified = True

                if self.geom_os_obj.delete_station(station_name):
                    self.is_model_modified = True

            elif cmd_key == "rename_IQ":
                IQ_old_name = cmd_param[0]
                IQ_new_name = cmd_param[1]

                print(f"rename_IQ: {IQ_old_name} -> {IQ_new_name}")

                if self.els_obj.rename_IQ(IQ_old_name, IQ_new_name):
                    self.replace_IQ_names({IQ_old_name: IQ_new_name})
                    self.is_model_modified = True

            elif cmd_key == "set_IQ_name":
                station_ref = cmd_param[0]
                ISO = cmd_param[1]
                IQ_new_name = cmd_param[2]

                print(f"set_IQ_name: {station_ref},{ISO} -> {IQ_new_name}")

                IQ_old_name = self.els_obj.set_IQ_name(station_ref, ISO, IQ_new_name)

                if IQ_old_name is not None:
                    self.replace_IQ_names({IQ_old_name: IQ_new_name})

                self.is_model_modified = True

            elif cmd_key == "set_IQ_def":
                IQ_name = cmd_param[0]
                ISO = cmd_param[1]
                IQ_type = cmd_param[2]
                IQ_unit = cmd_param[3]
                IQ_factor = cmd_param[4]

                IQ_attr = {
                    "ISO": ISO,
                    "type": IQ_type,
                    "unit": IQ_unit,
                    "factor": IQ_factor,
                }

                print(f"set_IQ_def: {IQ_name} {IQ_attr}")

                self.els_obj.set_IQ_attributes(IQ_name, IQ_attr)

                self.is_model_modified = True

            elif cmd_key == "set_IQ_station":
                IQ_name = cmd_param[0]
                IQ_station = cmd_param[1]

                print(f"set_IQ_station: {IQ_name} -> {IQ_station}")

                self.els_obj.set_IQ_station(IQ_name, IQ_station)

                self.is_model_modified = True

            elif cmd_key == "set_IQ_attr":
                IQ_name = cmd_param[0]
                attr_name = cmd_param[1]
                attr_value = cmd_param[2]

                print(f"set_IQ_attr: {IQ_name} {attr_name}={attr_value}")

                self.els_obj.set_IQ_attribute(IQ_name, attr_name, attr_value)

                self.is_model_modified = True

            elif cmd_key == "new_IQ":
                IQ_name = cmd_param[0]
                station_ref = cmd_param[1]
                ISO = cmd_param[2]
                IQ_type = cmd_param[3]
                IQ_unit = cmd_param[4]
                IQ_factor = cmd_param[5]

                station_obj = self.geom_os_obj.get_station_by_ref(station_ref)

                if station_obj is None:
                    print(f"Error during new_IQ: {IQ_name} - invalid station ref {station_ref}")
                    continue

                comp_key = station_obj.comp_key
                IQ_station = station_obj.name
                IQ_coords = station_obj.Coordinates
                station_desc = station_obj.Station_Description

                comp_name = self.get_component_data(comp_key, "name")
                SISO = self.get_component_data(comp_key, "SISO")
                comp_iaxis = self.get_component_data(comp_key, "iaxis")
                coord_axis = self.get_component_data(comp_key, "axis")

                IQ_attr = {
                    "name": IQ_name,
                    "comp_key": comp_key,
                    "comp_name": comp_name,
                    "SISO": SISO,
                    "ISO": ISO,
                    "type": IQ_type,
                    "station": IQ_station,
                    "station_desc": station_desc,
                    "unit": IQ_unit,
                    "factor": IQ_factor,
                    "coord": IQ_coords[comp_iaxis],
                    "coord_axis": coord_axis,
                }

                self.add_IQ(IQ_attr)

                print(f"new_IQ: {IQ_name}")

                self.is_model_modified = True

            elif cmd_key == "set_IQ_value":
                IQ_name = cmd_param[0]
                IQ_value = cmd_param[1]
                LRC_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"set_IQ_value: {IQ_name} = {IQ_value} ({LRC_name})")

                if self.set_IQ_value(IQ_name, LRC_name, IQ_value):
                    self.is_data_modified = True

            elif cmd_key == "set_IQ_DB_value":
                IQ_name = cmd_param[0]
                DB_value = cmd_param[1]
                LRC_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"set_IQ_DB_value: {IQ_name} = {DB_value} ({LRC_name})")

                if self.set_IQ_DB_value(IQ_name, LRC_name, DB_value):
                    self.is_data_modified = True

            elif cmd_key == "copy_IQ_value":
                IQ_name = cmd_param[0]
                from_IQ_name = cmd_param[1]
                factor = cmd_param[2]
                LRC_name = cmd_param[3]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"copy_IQ_value: {IQ_name} = {from_IQ_name} * {factor} ({LRC_name})")

                if self.copy_IQ_value(IQ_name, LRC_name, from_IQ_name, factor):
                    self.is_data_modified = True

            elif cmd_key == "scale_IQ_value":
                IQ_name = cmd_param[0]
                factor = cmd_param[1]
                LRC_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"scale_IQ_value: {IQ_name} * {factor} ({LRC_name})")

                if self.scale_IQ_value(IQ_name, LRC_name, factor):
                    self.is_data_modified = True

            elif cmd_key == "scale_IQ_value2":
                IQ_name = cmd_param[0]
                scale_IQ_name = cmd_param[1]
                factor = cmd_param[2]
                LRC_name = cmd_param[3]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"scale_IQ_value2: {IQ_name} * {scale_IQ_name} * {factor} ({LRC_name})")

                if self.scale_IQ_value2(IQ_name, LRC_name, scale_IQ_name, factor):
                    self.is_data_modified = True

            elif cmd_key == "convert_IQ_unit":
                IQ_name = cmd_param[0]
                unit = cmd_param[1]
                factor = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"convert_IQ_unit: {IQ_name} [{unit}] / {factor}")

                if self.convert_IQ_unit(IQ_name, unit, factor):
                    # self.is_model_modified = True
                    self.is_data_modified = True

            elif cmd_key == "delete_IQ":
                IQ_name = cmd_param[0]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"delete_IQ: {IQ_name}")

                if self.delete_IQ(IQ_name):
                    self.is_data_modified = True

            elif cmd_key == "mirror_LRC_data":
                mirror_def_type = cmd_param[0]
                mirror_def_file = cmd_param[1]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"mirror_LRC_data: {mirror_def_type} ({mirror_def_file})")

                if self.mirror_LRC_data(mirror_def_type, mirror_def_file):
                    self.is_data_modified = True

            elif cmd_key == "scale_LRC":
                factor = cmd_param[0]
                LRC_name = cmd_param[1]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"scale_LRC: {factor} ({LRC_name})")

                IQ_factor_vector = self.get_lrtc_IQ_factor_vector(factor)

                if IQ_factor_vector is None:
                    continue

                if self.scale_LRC(LRC_name, factor, IQ_factor_vector):
                    self.is_data_modified = True

            elif cmd_key == "scale_LRC_with_IQ_value":
                IQ_name = cmd_param[0]
                factor = cmd_param[1]
                LRC_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"scale_LRC_with_IQ_value: {IQ_name} * {factor} ({LRC_name})")

                IQ_factor_flags = self.get_lrtc_IQ_factor_flags()

                if IQ_factor_flags is None:
                    continue

                if self.scale_LRC_with_IQ_value(
                    LRC_name, IQ_name, factor, IQ_factor_flags
                ):
                    self.is_data_modified = True

            elif cmd_key == "rename_LRC":
                old_LRC_name = cmd_param[0]
                new_LRC_name = cmd_param[1]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"rename_LRC: <{old_LRC_name}> -> <{new_LRC_name}>")

                if self.rename_LRC(old_LRC_name, new_LRC_name):
                    self.is_data_modified = True

            elif cmd_key == "delete_LRC":
                LRC_name = cmd_param[0]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"delete_LRC: <{LRC_name}>")

                if self.delete_LRC(LRC_name):
                    self.is_data_modified = True

            elif cmd_key == "interpolate_IQ_values":
                IQ_name = cmd_param[0]
                IQ1_name = cmd_param[1]
                IQ2_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"interpolate_IQ_values: {IQ_name} -> ({IQ1_name} - {IQ2_name})")

                if self.interpolate_IQ_values(IQ_name, IQ1_name, IQ2_name):
                    self.is_data_modified = True

            elif cmd_key == "interpolate_env_values":
                IQ_name = cmd_param[0]
                IQ1_name = cmd_param[1]
                IQ2_name = cmd_param[2]

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"interpolate_env_values: {IQ_name} -> ({IQ1_name} - {IQ2_name})")

                if self.interpolate_env_values(IQ_name, IQ1_name, IQ2_name):
                    self.is_data_modified = True

            elif cmd_key == "set_LRC_mod_mode":
                LRC_mod_mode = cmd_param[0]

                print(f"set_LRC_mod_mode: {LRC_mod_mode}")

                self.set_LRC_mod_mode(LRC_mod_mode)

            elif cmd_key == "calc_LRE1D":
                if len(cmd_param) > 0:
                    calc_mode = cmd_param[0]
                else:
                    calc_mode = ""

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"calc_LRE1D: {calc_mode}")

                if self.calc_LRE1D(calc_mode):
                    self.is_data_modified = True

            elif cmd_key == "calc_LRE2D":
                if len(cmd_param) > 0:
                    calc_mode = cmd_param[0]
                else:
                    calc_mode = ""

                if self.is_model_modified:
                    self.reorganize_model_data()

                print(f"calc_LRE2D: {calc_mode}")

                if self.calc_LRE2D(calc_mode):
                    self.is_data_modified = True

            elif cmd_key == "exec_cmd_file":
                cmd_file_name = cmd_param[0]

                if not cmd_file_name.startswith("/"):
                    current_dir = os.getcwd()
                    tmp_file_name = os.path.join(current_dir, cmd_file_name)

                    if os.path.exists(tmp_file_name):
                        cmd_file_name = tmp_file_name
                    elif cmd_file_path is not None:
                        tmp_file_name = os.path.join(cmd_file_path, cmd_file_name)
                        if os.path.exists(tmp_file_name):
                            cmd_file_name = tmp_file_name

                if os.path.exists(cmd_file_name):
                    file_cmd_list = Path_Func.get_file_lines(cmd_file_name)
                else:
                    print(f"Error: mod_cmd_file not found ({cmd_file_name})")
                    continue

                file_cmd_list.append("%s,%s" % ("cmd_file_completed", cmd_file_name))

                cmd_list = file_cmd_list + cmd_list

                print(f"execute mod_cmd_file: {cmd_file_name}")

            elif cmd_key == "cmd_file_completed":
                cmd_file_name = cmd_param[0]

                print(f"completed mod_cmd_file: {cmd_file_name}")

            elif cmd_key == "complete_model":
                self.reorganize_model_data()

        # ---------------------------------------------------------------------

        self.LRC_mod_mode = ""

        self.new_LRC_names = {}

        if self.is_model_modified:
            self.reorganize_model_data()

    # ===========================================================================
    def reorganize_model_data(self):
        """ """
        # ---------------------------------------------------------------------

        self.geom_os_obj.reorganize_station_data()

        self.geom_os_obj.sort_stations(self.geom_basic_obj)

        IQ_names_modified = self.els_obj.reorganize_IQ_data()

        self.replace_IQ_names(IQ_names_modified)

        self.is_model_modified = False

    # ===========================================================================
    def replace_IQ_names(self, IQ_names_map):
        """ """
        # ---------------------------------------------------------------------

        IQ_names = []
        for IQ_name in self.IQ_names:
            IQ_new_name = IQ_name
            if IQ_name in IQ_names_map:
                IQ_new_name = IQ_names_map[IQ_name]
            IQ_names.append(IQ_new_name)
        self.IQ_names = IQ_names

        if self.lrtc_obj is not None:
            self.lrtc_obj.replace_IQ_names(IQ_names_map)

        if self.lre1d_obj is not None:
            self.lre1d_obj.replace_IQ_names(IQ_names_map)

        if self.lre2d_obj is not None:
            self.lre2d_obj.replace_IQ_names(IQ_names_map)

    # ===========================================================================
    def set_LRC_mod_mode(self, LRC_mod_mode):
        """
        sets the mode for LRC data modifications
        - modify    = just modify the LRCs
        - rename    = rename + modify the LRCs
        - duplicate = duplicate + rename + modify LRCs
        """
        # ---------------------------------------------------------------------

        if LRC_mod_mode in ("modify", "rename", "duplicate"):
            self.LRC_mod_mode = LRC_mod_mode
        else:
            print(f"invalid LRC_mod_mode: {LRC_mod_mode} -> set to 'modify'")
            self.LRC_mod_mode = "modify"

        self.new_LRC_names = {}

    # =========================================================================
    def mirror_LRC_data(self, mirror_def_type, mirror_def_file):
        """
        duplicates all LRCs and applies IQ mirror factors according to the mirror def data
        """
        if not self.load_LRC_mirror_def_data(mirror_def_type, mirror_def_file):
            return False

        if not self.LRC_mirror_def_data or len(self.LRC_mirror_def_data) == 0:
            return False

        # ---------------------------------------------------------------------

        self.remove_env_data()

        lrtc_IQ_names = self.lrtc_obj.get_IQ_names()

        mirror_def_index_data = []

        for IQ_mirror_def in self.LRC_mirror_def_data:
            IQ1_name = IQ_mirror_def[0]
            IQ2_name = IQ_mirror_def[1]
            factor = IQ_mirror_def[2]

            if IQ1_name in lrtc_IQ_names and IQ2_name in lrtc_IQ_names:
                IQ1_index = lrtc_IQ_names.index(IQ1_name)
                IQ2_index = lrtc_IQ_names.index(IQ2_name)
                mirror_def_index_data.append([IQ1_index, IQ2_index, factor])

        # ---------------------------------------------------------------------

        org_LRC_names = self.lrtc_obj.get_LRC_names()

        for ind_LRC, org_LRC_name in enumerate(org_LRC_names):
            new_LRC_name = self.get_mirror_LRC_name(org_LRC_name)

            if new_LRC_name is None:
                continue

            print(f"LRC mirroring [{ind_LRC + 1}]: {org_LRC_name} -> {new_LRC_name}")

            mass_case = self.lrtc_obj.get_LRC_MassCase(org_LRC_name)
            org_LRC_row = self.lrtc_obj.get_LRC_row(org_LRC_name)
            org_LRC_file = self.get_LRC_file_name(org_LRC_name)

            new_LRC_row = org_LRC_row[:]

            for IQ_mirror_def in mirror_def_index_data:
                new_IQ_index = IQ_mirror_def[0]
                org_IQ_index = IQ_mirror_def[1]
                mirr_factor = IQ_mirror_def[2]

                if org_LRC_row[org_IQ_index] is not None:
                    new_LRC_row[new_IQ_index] = org_LRC_row[org_IQ_index] * mirr_factor

            self.lrtc_obj.add_LRC_row(new_LRC_name, mass_case, new_LRC_row)

            self.add_LRC_name(new_LRC_name, mass_case, org_LRC_file)
            self.new_LRC_names[org_LRC_name] = new_LRC_name

            if self.SRC_names is not None and org_LRC_name in self.SRC_names:
                SRC_name = self.SRC_names[org_LRC_name]
                self.set_SRC_name(new_LRC_name, SRC_name)

        print("LRC mirroring completed")

        return True

    # =========================================================================
    def set_IQ_DB_value(self, IQ_name, LRC_name, DB_value):
        """
        sets a new IQ value (in DB units) for a given IQ name and LRC name
        """
        if DB_value is not None:
            try:
                DB_value = float(DB_value)
            except (TypeError, ValueError):
                print(f"set_IQ_DB_value: invalid DB value {IQ_name} = {DB_value} ({LRC_name})")
                return False
        else:
            print(f"set_IQ_DB_value: invalid DB value {IQ_name} = {DB_value} ({LRC_name})")
            return False

        IQ_value = DB_value / self.get_IQ_factor(IQ_name)  # convert from DB units to SI units

        return self.set_IQ_value(IQ_name, LRC_name, IQ_value)

    # =========================================================================
    def set_IQ_value(self, IQ_name, LRC_name, IQ_value):
        """
        sets a new IQ value for a given IQ name and LRC name
        """

        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.set_IQ_value(IQ_name, LRC_name, IQ_value)
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"set_IQ_value: invalid LRC name {IQ_name} = {IQ_value} ({LRC_name})")
            return False

        if IQ_value is not None:
            try:
                IQ_value = float(IQ_value)
            except (TypeError, ValueError):
                print(f"set_IQ_value: invalid IQ value {IQ_name} = {IQ_value} ({LRC_name})")
                return False
        else:
            print(f"set_IQ_value: invalid IQ value {IQ_name} = {IQ_value} ({LRC_name})")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        self.lrtc_obj.set_IQ_value(IQ_name, LRC_name, IQ_value)

        return True

    # =========================================================================
    def copy_IQ_value(self, IQ_name, LRC_name, from_IQ_name, factor):
        """
        copies a IQ value for a given IQname and LRC name from another IQ value * factor
        """

        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.copy_IQ_value(IQ_name, LRC_name, from_IQ_name, factor)
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"copy_IQ_value: invalid LRC name {IQ_name} = {from_IQ_name} * {factor} ({LRC_name})")
            return False

        if factor is None:
            factor = 1.0

        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"copy_IQ_value: invalid factor {IQ_name} = {from_IQ_name} * {factor} ({LRC_name})")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        IQ_value = self.get_IQ_value(from_IQ_name, LRC_name)

        if IQ_value is not None:
            IQ_value *= factor

        self.lrtc_obj.set_IQ_value(IQ_name, LRC_name, IQ_value)

        return True

    # =========================================================================
    def scale_IQ_value(self, IQ_name, LRC_name, factor):
        """
        scales a IQ value for a given IQ name and LRC name
        """

        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.scale_IQ_value(IQ_name, LRC_name, factor)
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"scale_IQ_value: invalid LRC name {IQ_name} = {factor} ({LRC_name})")
            return False

        if factor is None:
            print(f"scale_IQ_value: invalid factor {IQ_name} = {factor} ({LRC_name})")
            return False

        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"scale_IQ_value: invalid factor {IQ_name} = {factor} ({LRC_name})")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        IQ_value = self.get_IQ_value(IQ_name, LRC_name)

        if IQ_value is None:
            return False

        self.lrtc_obj.set_IQ_value(IQ_name, LRC_name, IQ_value * factor)

        return True

    # =========================================================================
    def scale_IQ_value2(self, IQ_name, LRC_name, scale_IQ_name, factor):
        """
        scales a IQ value for a given IQ name and LRC name with the value of another IQ
        """

        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.scale_IQ_value2(IQ_name, LRC_name, scale_IQ_name, factor)
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if LRC_name not in self.LRC_index:
            print(f"scale_IQ_value2: invalid LRC name {IQ_name} = {factor} ({LRC_name})")
            return False

        if factor is None:
            factor = 1.0
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"scale_IQ_value2: invalid factor {IQ_name} = {factor} ({LRC_name})")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        scale_IQ_value = self.get_IQ_value(scale_IQ_name, LRC_name)
        if scale_IQ_value is None:
            print(f"scale_IQ_value2: factor IQ has no value {scale_IQ_name} ({LRC_name})")
            return False

        IQ_value = self.get_IQ_value(IQ_name, LRC_name)

        if IQ_value is None:
            return False

        self.lrtc_obj.set_IQ_value(
            IQ_name, LRC_name, IQ_value * scale_IQ_value * factor
        )

        return True

    # =========================================================================
    def convert_IQ_unit(self, IQ_name, unit, factor):
        """
        converts the unit and scales all the IQ values for a given IQ name
        """
        if factor is None:
            print(f"convert_IQ_unit: invalid factor {IQ_name} [{unit}] / {factor}")
            return False

        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"convert_IQ_unit: invalid factor {IQ_name} [{unit}] / {factor}")
            return False

        self.els_obj.set_IQ_attribute(IQ_name, "unit", unit)
        self.els_obj.set_IQ_attribute(IQ_name, "factor", factor)

        self.remove_env_data()

        LRC_names = self.LRC_names[:]

        for LRC_name in LRC_names:
            mod_LRC_name = self.get_mod_LRC_name(LRC_name)

            old_value = self.get_IQ_value(IQ_name, mod_LRC_name)

            if old_value is None:
                continue

            IQ_value = old_value / factor

            self.lrtc_obj.set_IQ_value(IQ_name, mod_LRC_name, IQ_value)

        return True

    # =========================================================================
    def delete_IQ(self, IQ_name):
        """
        deletes an IQ and all related data
        """
        self.els_obj.delete_IQ(IQ_name)

        if self.lrtc_obj is not None:
            self.lrtc_obj.delete_IQ(IQ_name)

        if self.lre1d_obj is not None:
            self.lre1d_obj.delete_IQ(IQ_name)

        if self.lre2d_obj is not None:
            self.lre2d_obj.delete_IQ(IQ_name)

        if IQ_name in self.IQ_names:
            self.IQ_names.remove(IQ_name)

        return True

    # ===========================================================================
    def get_lrtc_IQ_factor_vector(self, factor):
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"get_lrtc_IQ_factor_vector: invalid factor {factor}")
            return None

        IQ_factor_flags = self.get_lrtc_IQ_factor_flags()

        IQ_factor_vector = [factor if flag else 1.0 for flag in IQ_factor_flags]

        return IQ_factor_vector

    # ===========================================================================
    def get_lrtc_IQ_factor_flags(self):
        IQ_names = self.lrtc_obj.get_IQ_names()

        IQ_factor_flags = [True] * len(IQ_names)

        for ind_IQ, IQ_name in enumerate(IQ_names):
            comp_key = self.get_IQ_attribute(IQ_name, "comp_key")

            if comp_key == "AC":
                IQ_unit = self.get_IQ_unit(IQ_name)
                if IQ_unit.upper() not in ("N", "NM", "DAN", "DANM"):
                    IQ_factor_flags[ind_IQ] = False

        return IQ_factor_flags

    # =========================================================================
    def scale_LRC(self, LRC_name, factor, IQ_factor_vector=None):
        """
        scales all IQ values of a given LRC by factor
        """
        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.scale_LRC(LRC_name, factor, IQ_factor_vector)
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if self.lrtc_obj is None:
            print(f"scale_LRC: no LRC data available {LRC_name}")
            return False

        if LRC_name not in self.LRC_index:
            print(f"scale_LRC: invalid LRC name {LRC_name}")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        if IQ_factor_vector is None:
            IQ_factor_vector = self.get_lrtc_IQ_factor_vector(factor)

        if IQ_factor_vector is None:
            return False

        self.lrtc_obj.multiply_LRC_row(LRC_name, IQ_factor_vector)

        return True

    # =========================================================================
    def scale_LRC_with_IQ_value(
        self, LRC_name, scale_IQ_name, factor, IQ_factor_flags=None
    ):
        """
        scales all IQ values of a given LRC by the value of a given IQ * factor
        """

        if LRC_name == "*":
            for LRC_name in self.get_LRC_names():
                self.scale_LRC_with_IQ_value(
                    LRC_name, scale_IQ_name, factor, IQ_factor_flags
                )
            return True

        # ---------------------------------------------------------------------

        LRC_name = LRC_name.ljust(40)[:40]

        if self.lrtc_obj is None:
            print(f"scale_LRC_with_IQ_value: no LRC data available {LRC_name}")
            return False

        if LRC_name not in self.LRC_index:
            print(f"scale_LRC_with_IQ_value: invalid LRC name {LRC_name}")
            return False

        try:
            factor = float(factor)
        except (TypeError, ValueError):
            print(f"scale_LRC_with_IQ_value: invalid factor {factor}")
            return False

        self.remove_env_data()

        LRC_name = self.get_mod_LRC_name(LRC_name)

        scale_IQ_value = self.get_IQ_value(scale_IQ_name, LRC_name)
        if scale_IQ_value is None:
            print(f"scale_LRC_with_IQ_value: factor IQ has no value {scale_IQ_name} ({LRC_name})")
            return False

        if IQ_factor_flags is None:
            IQ_factor_flags = self.get_lrtc_IQ_factor_flags()

        IQ_factor_vector = [
            scale_IQ_value * factor if flag else 1.0 for flag in IQ_factor_flags
        ]

        self.lrtc_obj.multiply_LRC_row(LRC_name, IQ_factor_vector)

        return True

    # =========================================================================
    def interpolate_IQ_values(self, IQ_name, IQ1_name, IQ2_name):
        """
        calculates new IQ values for all LRCs for a given IQ name by interpolation
        """
        if self.lrtc_obj is None:
            print("interpolate_IQ_values: no LRC data available")
            return False

        IQ_obj = self.els_obj.get_IQ_def(IQ_name)
        IQ1_obj = self.els_obj.get_IQ_def(IQ1_name)
        IQ2_obj = self.els_obj.get_IQ_def(IQ2_name)

        if IQ_obj is None or IQ1_obj is None or IQ2_obj is None:
            print(f"interpolate_IQ_values: invalid IQ name(s) {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        comp_key = IQ_obj.comp_key
        comp_key1 = IQ1_obj.comp_key
        comp_key2 = IQ2_obj.comp_key

        if comp_key != comp_key1 or comp_key != comp_key2:
            print(f"interpolate_IQ_values: IQs are not from same component {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        IQ_coord = IQ_obj.coord
        IQ1_coord = IQ1_obj.coord
        IQ2_coord = IQ2_obj.coord

        if IQ_coord is None or IQ1_coord is None or IQ2_coord is None:
            print(f"interpolate_IQ_values: incomplete coordinate values {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        # -----------------------------------------------------------------------

        self.remove_env_data()

        LRC_names = self.LRC_names[:]

        for LRC_name in LRC_names:
            mod_LRC_name = self.get_mod_LRC_name(LRC_name)

            IQ1_value = self.get_IQ_value(IQ1_name, mod_LRC_name)
            IQ2_value = self.get_IQ_value(IQ2_name, mod_LRC_name)
            IQ_value = None

            if IQ1_value is not None and IQ2_value is not None:
                IQ_value = np.interp(
                    IQ_coord,
                    [IQ1_coord, IQ2_coord],
                    [IQ1_value, IQ2_value],
                    left=IQ1_value,
                    right=IQ2_value,
                )

            if IQ_value is not None:
                self.lrtc_obj.set_IQ_value(IQ_name, mod_LRC_name, IQ_value)

        return True

    # =========================================================================
    def interpolate_env_values(self, IQ_name, IQ1_name, IQ2_name):
        """
        calculates new IQ values for 2 new  envelope LRCs by interpolation
        """
        IQ_obj = self.els_obj.get_IQ_def(IQ_name)
        IQ1_obj = self.els_obj.get_IQ_def(IQ1_name)
        IQ2_obj = self.els_obj.get_IQ_def(IQ2_name)

        if IQ_obj is None or IQ1_obj is None or IQ2_obj is None:
            print(f"interpolate_env_values: invalid IQ name(s) {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        comp_key = IQ_obj.comp_key
        comp_key1 = IQ1_obj.comp_key
        comp_key2 = IQ2_obj.comp_key

        if comp_key != comp_key1 or comp_key != comp_key2:
            print(f"interpolate_env_values: IQs are not from same component {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        IQ_coord = IQ_obj.coord
        IQ1_coord = IQ1_obj.coord
        IQ2_coord = IQ2_obj.coord

        if IQ_coord is None or IQ1_coord is None or IQ2_coord is None:
            print(f"interpolate_env_values: incomplete coordinate values {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        # -----------------------------------------------------------------------

        IQ1_min_max_values = self.get_IQ_min_max_values(IQ1_name)
        IQ2_min_max_values = self.get_IQ_min_max_values(IQ2_name)

        if IQ1_min_max_values is None or IQ2_min_max_values is None:
            print(f"interpolate_env_values: missing IQ values for interpolation {IQ_name} -> ({IQ1_name} - {IQ2_name})")
            return False

        [
            min_IQ1_value,
            max_IQ1_value,
            _min_LRC1_name,
            _max_LRC1_name,
        ] = IQ1_min_max_values
        [
            min_IQ2_value,
            max_IQ2_value,
            _min_LRC2_name,
            _max_LRC2_name,
        ] = IQ2_min_max_values

        min_IQ_value = np.interp(
            IQ_coord,
            [IQ1_coord, IQ2_coord],
            [min_IQ1_value, min_IQ2_value],
            left=min_IQ1_value,
            right=min_IQ2_value,
        )

        max_IQ_value = np.interp(
            IQ_coord,
            [IQ1_coord, IQ2_coord],
            [max_IQ1_value, max_IQ2_value],
            left=max_IQ1_value,
            right=max_IQ2_value,
        )

        # -----------------------------------------------------------------------

        timestamp = time.strftime("%Y%m%d%H%M%S")

        new_min_LRC_name = IQ_name + "_min1_" + timestamp
        new_max_LRC_name = IQ_name + "_max1_" + timestamp

        new_min_LRC_name = new_min_LRC_name.ljust(40)[:40]
        new_max_LRC_name = new_max_LRC_name.ljust(40)[:40]

        # -----------------------------------------------------------------------

        self.remove_env_data()

        IQ_names = self.lrtc_obj.get_IQ_names()
        nIQs = len(IQ_names)
        new_max_LRC_row = [None] * nIQs
        new_min_LRC_row = [None] * nIQs

        self.lrtc_obj.add_LRC_row(new_min_LRC_name, "", new_min_LRC_row)
        self.lrtc_obj.add_LRC_row(new_max_LRC_name, "", new_max_LRC_row)

        self.lrtc_obj.set_IQ_value(IQ_name, new_min_LRC_name, min_IQ_value)
        self.lrtc_obj.set_IQ_value(IQ_name, new_max_LRC_name, max_IQ_value)

        return True

    # =========================================================================
    def calc_LRE1D(self, calc_mode):
        """
        generates/re-calculates the LRE1D data
        """
        if not self.has_LRC_values:
            print("calc_LRE1D: no LRC data available")
            return False

        if calc_mode == "LRE1D" and (
            self.lre1d_obj is None and self.env1d_IQ_rank_lists is None
        ):
            print("calc_LRE1D: no LRE1D data available")
            return False

        # -----------------------------------------------------------------------

        calc_IQ_names = []

        if calc_mode == "ELS":
            for IQ_name in self.IQ_names:
                if self.els_obj.get_IQ_attribute(IQ_name, "Selection_flag"):
                    calc_IQ_names.append(IQ_name)

        elif calc_mode == "LRE1D":
            if self.lre1d_obj is None:
                for IQ_name in self.IQ_names:
                    if IQ_name in self.env1d_IQ_rank_lists:
                        calc_IQ_names.append(IQ_name)
            else:
                for IQ_name in self.IQ_names:
                    if self.lre1d_obj.has_IQ(IQ_name):
                        calc_IQ_names.append(IQ_name)

        elif calc_mode == "all":
            calc_IQ_names = self.IQ_names

        else:
            print(f"calc_LRE1D: invalid calc_mode {calc_mode}")
            return False

        n_calc_IQs = len(calc_IQ_names)

        if n_calc_IQs == 0:
            return False

        # -----------------------------------------------------------------------

        if self.lre1d_obj is None:
            lre1d_obj = LRE1D_data()
            lre1d_obj.init_data(
                Aircraft=self.get_info("Aircraft"),
                Project=self.get_info("Project"),
                Date=self.get_info("Date"),
                DB_name=self.get_info("DB_name"),
                LimitUltimate=self.get_info("LimUlt"),
            )
        else:
            lre1d_obj = self.lre1d_obj

        # ignore the LRE1D object during the calculation -> get min/max values from LRC data

        self.lre1d_obj = None
        self.has_env1d_data = False

        # -----------------------------------------------------------------------

        for ind_IQ, IQ_name in enumerate(calc_IQ_names):
            IQ_min_max_values = self.get_IQ_min_max_values(IQ_name)

            if IQ_min_max_values is None:
                continue

            min_IQ_value = IQ_min_max_values[0]
            max_IQ_value = IQ_min_max_values[1]
            min_LRC_name = IQ_min_max_values[2]
            max_LRC_name = IQ_min_max_values[3]

            print(f"calc_LRE1D[{ind_IQ + 1}/{n_calc_IQs}]: {IQ_name} (min={min_IQ_value} ; max={max_IQ_value})")

            lre1d_obj.add_IQ_rank(IQ_name, "min1", min_IQ_value, min_LRC_name)
            lre1d_obj.add_IQ_rank(IQ_name, "max1", max_IQ_value, max_LRC_name)

        # -----------------------------------------------------------------------

        self.lre1d_obj = lre1d_obj
        self.has_env1d_data = True
        self.env1d_IQ_rank_lists = None

        return True

    # =========================================================================
    def calc_LRE2D(self, calc_mode):
        """
        generates/re-calculates the LRE2D data
        """
        if self.lrtc_obj is None:
            print("calc_LRE2D: no LRC data available")
            return False

        if calc_mode == "LRE2D" and (
            self.lre2d_obj is None and self.env2d_dict is None
        ):
            print("calc_LRE2D: no LRE2D data available")
            return False

        lre2d_obj = LRE2D_data()
        lre2d_obj.init_data(
            Aircraft=self.get_info("Aircraft"),
            Project=self.get_info("Project"),
            Date=self.get_info("Date"),
            DB_name=self.get_info("DB_name"),
            LimitUltimate=self.get_info("LimUlt"),
        )

        # -----------------------------------------------------------------------

        lrtc_IQ_names = self.lrtc_obj.get_IQ_names()

        calc_CIQs = []

        if calc_mode == "ELS":
            corr_data = self.els_obj.get_plot_data2("CQ", 0)

            if corr_data is None:
                return False

            for IQ_name in corr_data["IQ_names"]:
                formula = self.els_obj.get_IQ_attribute(IQ_name, "Formula")
                IQ1_name = None
                IQ2_name = None

                ind1 = formula.find("{")
                ind2 = formula.find("}")
                if ind1 > 0 and ind2 > 0:
                    IQ_name = formula[ind1 + 1: ind2].strip()
                    formula = formula[ind2 + 1:]
                    if IQ_name in lrtc_IQ_names:
                        IQ1_name = IQ_name

                ind1 = formula.find("{")
                ind2 = formula.find("}")
                if ind1 > 0 and ind2 > 0:
                    IQ_name = formula[ind1 + 1: ind2].strip()
                    if IQ_name in lrtc_IQ_names:
                        IQ2_name = IQ_name

                if IQ1_name and IQ2_name:
                    calc_CIQs.append([IQ1_name, IQ2_name])

        elif calc_mode == "LRE2D":
            if self.lre2d_obj is None:
                env2d_dict = self.env2d_dict
            else:
                env2d_dict = self.lre2d_obj.get_env_dict()
            for IQ1_name in env2d_dict.keys():
                for IQ2_name in env2d_dict[IQ1_name].keys():
                    calc_CIQs.append([IQ1_name, IQ2_name])
        else:
            print(f"calc_LRE2D: invalid calc_mode {calc_mode}")
            return False

        n_calc_CIQs = len(calc_CIQs)

        if n_calc_CIQs == 0:
            return False

        # -----------------------------------------------------------------------

        LRC_names = self.lrtc_obj.get_LRC_names()

        for ind_CIQ, IQ_names in enumerate(calc_CIQs):
            IQ1_name = IQ_names[0]
            IQ2_name = IQ_names[1]

            print(f"calc_LRE2D[{ind_CIQ + 1}/{n_calc_CIQs}]: {IQ1_name} -> {IQ2_name}")

            LRC_points = []

            for ind_LRC, LRC_name in enumerate(LRC_names):
                IQ1_value = self.lrtc_obj.get_IQ_value(IQ1_name, LRC_name)
                IQ2_value = self.lrtc_obj.get_IQ_value(IQ2_name, LRC_name)

                if IQ1_value is None or IQ2_value is None:
                    continue

                point = LRC_point(IQ1_value, IQ2_value, LRC_name)
                LRC_points.append(point)
                point.index = len(LRC_points) - 1

            if len(LRC_points) < 3:
                continue

            hull_start_index = Convex_Hull(LRC_points)

            if hull_start_index is None:
                continue

            env2d_points = []
            round_count = 0
            hull_point = LRC_points[hull_start_index]

            while True:
                env2d_points.append(hull_point)
                if hull_point.index == hull_start_index:
                    if round_count == 0:
                        round_count += 1
                    else:
                        break
                hull_point = LRC_points[hull_point.next]

            CIQ_data = {"IQ_names": IQ_names,
                        "IQ_values": [],
                        "LRC_names": [],
                        "mass_cases": [],
                        "lrtc_files": []
                        }

            for env2d_point in env2d_points:
                mass_case = self.lrtc_obj.get_LRC_MassCase(env2d_point.LRC_name)
                CIQ_data["IQ_values"].append([env2d_point.x, env2d_point.y])
                CIQ_data["LRC_names"].append(env2d_point.LRC_name)
                CIQ_data["mass_cases"].append(mass_case)
                CIQ_data["lrtc_files"].append("")

            lre2d_obj.add_CIQ(CIQ_data)

        # -----------------------------------------------------------------------

        self.lre2d_obj = lre2d_obj
        self.has_env2d_data = True
        self.env2d_dict = None

        return True

    # ===========================================================================
    def clone_data_set(self):
        """
        creates a copy of the Data_set in memory
        """
        copy = ATLAS_Data_Set(geom_basic="")

        copy.name = "copy_of>" + self.name
        copy.file_name = "copy_of>" + self.file_name

        copy.set_info("Aircraft", self.get_info("Aircraft"))
        copy.set_info("Project", self.get_info("Project"))
        copy.set_info("Date", self.get_info("Date"))
        copy.set_info("DB_name", self.get_info("DB_name"))
        copy.set_info("LimUlt", self.get_info("LimUlt"))
        copy.set_info("Units", self.get_info("Units"))

        # -----------------------------------------------------------------------

        copy.geom_basic_obj = GEOM_Basic_data()
        copy.geom_os_obj = GEOM_OS_data()
        copy.els_obj = ELS_data()

        copy.geom_basic_obj.set_ads_data(self.geom_basic_obj.get_ads_data())
        copy.geom_os_obj.set_ads_data(self.geom_os_obj.get_ads_data())
        copy.els_obj.set_csv_data(self.els_obj.get_csv_data())

        copy.complete_model_set()

        # -----------------------------------------------------------------------

        if self.has_LRC_values:
            if self.LRC_matrix_type == "sparse":
                copy.lrtc_obj = LRTC_sparse_data()
            else:
                copy.lrtc_obj = LRTC_data()

            copy.has_LRC_values = True
            copy.LRC_matrix_type = self.LRC_matrix_type

            copy.lrtc_obj.init_data(
                Aircraft=self.get_info("Aircraft"),
                Project=self.get_info("Project"),
                Date=self.get_info("Date"),
                DB_name=self.get_info("DB_name"),
                LimitUltimate=self.get_info("LimUlt"),
            )

            (
                IQ_names,
                LRC_names,
            ) = self.get_LRC_matrix_names()  # IQ and LRC names of the LRC value matrix

            for IQ_name in IQ_names:
                copy.add_IQ_name(IQ_name)

            copy.lrtc_obj.set_IQ_names(IQ_names)

            for LRC_name in LRC_names:
                mass_case = self.get_LRC_MassCase(LRC_name)
                LRC_row = self.get_LRC_row(LRC_name)
                copy.lrtc_obj.add_LRC_row(LRC_name, mass_case, LRC_row)
                copy.add_LRC_name(LRC_name, mass_case)

            copy.LRC_file_list.append("")
            copy.LRC_file_index[""] = 0

        # ---------------------------------------------------------------------

        if self.lre1d_obj:
            copy.lre1d_obj = LRE1D_data()
            copy.has_env1d_data = True

            lre1d_IQ_names = self.lre1d_obj.get_IQ_names()

            for IQ_name in lre1d_IQ_names:
                copy.add_IQ_name(IQ_name)

                rank_desc_list = self.lre1d_obj.get_IQ_rank_desc_list(IQ_name)

                for rank_desc in rank_desc_list:
                    (
                        IQ_value,
                        LRC_name,
                        mass_case,
                        lrtc_file,
                    ) = self.lre1d_obj.get_IQ_rank_data(IQ_name, rank_desc)
                    copy.lre1d_obj.add_IQ_rank(IQ_name, rank_desc, IQ_value, LRC_name)
                    copy.add_LRC_name(LRC_name, mass_case)

        # ---------------------------------------------------------------------

        if self.lre2d_obj:
            copy.lre2d_obj = LRE2D_data()
            copy.has_env2d_data = True

            lre2d_env_dict = self.lre2d_obj.get_env_dict()

            for IQ1_name in lre2d_env_dict.keys():
                copy.add_IQ_name(IQ1_name)

                for IQ2_name in lre2d_env_dict[IQ1_name].keys():
                    copy.add_IQ_name(IQ2_name)

                    CIQ_data = self.lre2d_obj.get_env_data(IQ1_name, IQ2_name)

                    CIQ_data["IQ_names"] = [IQ1_name, IQ2_name]

                    copy.lre2d_obj.add_CIQ(CIQ_data)

                    for i_corner in range(CIQ_data["n_corners"]):
                        LRC_name = CIQ_data["LRC_names"][i_corner]
                        mass_case = CIQ_data["mass_cases"][i_corner]
                        copy.add_LRC_name(LRC_name, mass_case)

        # ---------------------------------------------------------------------

        for LRC_name in copy.LRC_names:
            copy.LRC_name_file_index[LRC_name] = 0

        for key in self.data_source_files:
            copy.data_source_files[key] = self.data_source_files[key]

        if self.SRC_names is not None:
            copy.SRC_names = self.SRC_names.copy()

        # ---------------------------------------------------------------------

        return copy

    # ===========================================================================
    def print_info(self):
        """
        prints the object info data
        """
        print(f"File: {self.file_name}")

        for name in self.info.keys():
            print(f"{name}: {self.info[name]}")

    # ===========================================================================
    def get_info(self, attr_name=None):
        """
        returns the file header info data
        """
        if attr_name is None:
            return self.info
        elif attr_name in self.info:
            return self.info[attr_name]
        elif attr_name in info_name_map:
            info_name = info_name_map[attr_name]
            return self.info[info_name]
        else:
            return ""

    # ===========================================================================
    def set_info_data(self, info_data):
        """
        sets the file header info data
        """
        for attr_name in info_data.keys():
            self.set_info(attr_name, info_data[attr_name])

    # ===========================================================================
    def set_info(self, attr_name, attr_value):
        """
        sets a value in the file header info data
        """
        info_name = attr_name

        if info_name in info_name_map:
            info_name = info_name_map[info_name]

        self.info[info_name] = attr_value

    # ===========================================================================
    def get_file_name(self):
        """
        returns the name of the data set
        """
        return self.file_name

    # =========================================================================
    def get_Project_name(self):
        """
        returns the Project name from the best fit data source file
        """
        return self.get_info("Project")

    # =========================================================================
    def get_DB_name(self):
        """
        returns the DB name from the best fit data source file
        """
        return self.get_info("DB_name")

    # ===========================================================================
    def get_LRC_names(self):
        """
        returns the list of all LRC names
        """
        return self.LRC_names[:]

    # ===========================================================================
    def get_LRC_index(self, LRC_name):
        """
        returns the LRC row index for a given LRC name
        """
        if LRC_name in self.LRC_index:
            return self.LRC_index[LRC_name]
        else:
            return None

    # ===========================================================================
    def has_LRC(self, LRC_name):
        """
        true if the given LRC name exists in the LRC data
        """
        if LRC_name in self.LRC_index:
            return True
        else:
            return False

    # =========================================================================
    def get_LRC_row(self, LRC_name):
        """
        returns an LRC row for a given LRC name
        """
        if self.has_LRC_values and LRC_name in self.LRC_index:
            return self.lrtc_obj.get_LRC_row(LRC_name)

        return None

    # =========================================================================
    def get_LRC_matrix_names(self):
        """
        returns the list of IQ names and LRC names of the LRC values matrix
        """
        if self.has_LRC_values:
            IQ_names = self.lrtc_obj.get_IQ_names()
            LRC_names = self.lrtc_obj.get_LRC_names()
            return IQ_names, LRC_names

        return None

    # ===========================================================================
    def get_LRC_DB_name(self, LRC_name):
        """
        returns the DB_name for a given LRC name
        """
        if self.has_LRC_values and LRC_name in self.LRC_index:
            return self.lrtc_obj.get_LRC_DB_name(LRC_name)

        return ""

    # ===========================================================================
    def get_LRC_DB_code(self, LRC_name):
        """
        returns the DB_code for a given LRC name
        """
        if self.has_LRC_values and LRC_name in self.LRC_index:
            return self.lrtc_obj.get_LRC_DB_code(LRC_name)

        return ""

    # ===========================================================================
    def get_LRC_file_name(self, LRC_name):
        """
        returns the LRTC file name for a given LRC name
        """
        if self.has_LRC_values and LRC_name in self.LRC_index:
            if LRC_name in self.LRC_name_file_index:
                file_index = self.LRC_name_file_index[LRC_name]
                if file_index is not None:
                    return self.LRC_file_list[file_index]

        return ""

    # ===========================================================================
    def get_MassCase_names(self):
        """
        returns the list of MassCase names for all LRCs
        """
        return self.MassCase_names[:]

    # ===========================================================================
    def get_LRC_MassCase(self, LRC_name):
        """
        returns the MassCase name for a given LRC name
        """
        if LRC_name in self.LRC_index:
            ind_LRC = self.LRC_index[LRC_name]
            return self.MassCase_names[ind_LRC]

        return None

    # ===========================================================================
    def get_MassCase_LRC_names(self, mass_case):
        """
        returns the LRC names related to the given MassCase name
        """
        MassCase_names = []
        for ind_LRC, LRC_name in enumerate(self.LRC_names):
            if self.MassCase_names[ind_LRC].upper() == mass_case.upper():
                MassCase_names.append(LRC_name)

        return MassCase_names

    # ===========================================================================
    def get_SRC_names(self):
        """
        returns the list of all SRC names
        """
        if self.SRC_names is None:
            return None

        return [self.SRC_names.get(LRC_name, "") for LRC_name in self.LRC_names]

    # ===========================================================================
    def get_SRC_name(self, LRC_name):
        """
        returns the SRC name for the given LRC name
        """
        if self.SRC_names is None:
            return None

        if LRC_name in self.SRC_names:
            return self.SRC_names[LRC_name]
        else:
            return ""

    # ===========================================================================
    def get_extra_datasets(self):
        """
        return the list of extra datasets (without "_obj") with Loads data (apart from lrtc) existing in the atlas_obj
        """

        extra_ds = []

        for attr_name in dir(self):
            if not attr_name.endswith('_obj'):
                continue

            header_attr_name = attr_name.replace("_obj", "_header")

            if hasattr(self, attr_name) and hasattr(self, header_attr_name):
                if not attr_name.endswith("_header"):
                    extra_ds.append(attr_name.replace("_obj", ""))

        return extra_ds

    # ===========================================================================
    def get_IQ_names(self):
        """
        returns the list of IQ names
        """
        # -----------------------------------------------------------------------

        return self.IQ_names[:]

    # ===========================================================================
    def get_IQ_names_at_station(self, station_name):
        """
        get the list of IQ_names from all IQs at a given station
        """
        # -----------------------------------------------------------------------

        if self.has_model_data:
            return self.els_obj.get_IQ_names_at_station(station_name)

        return None

    # ===========================================================================
    def get_IQ_station(self, IQ_name):
        """
        returns the station name for a given IQ name
        """
        if self.has_model_data and IQ_name in self.IQ_names:
            return self.els_obj.get_IQ_station(IQ_name)

        return None

    # ===========================================================================
    def get_IQ_station_description(self, IQ_name):
        """
        returns the station description string for a given IQ name
        """
        IQ_station = self.get_IQ_station(IQ_name)
        if IQ_station:
            return self.geom_os_obj.get_station_description(IQ_station)

        return None

    # ===========================================================================
    def get_IQ_station_coordinates(self, IQ_name, axis_mask=None):
        """
        returns a list of coordinate values for a given output station
        (axis values selection and sequence is defined in axis_mask e.g. 'ZY')
        """
        IQ_station = self.get_IQ_station(IQ_name)
        if IQ_station:
            return self.geom_os_obj.get_station_coordinates(IQ_station, axis_mask)

        return None

    # ===========================================================================
    def get_IQ_station_index(self, IQ_name):
        """
        returns the station index for a given IQ name in its component/IQ_type list
        """
        return self.els_obj.get_IQ_attribute(IQ_name, "station_index")

    # ===========================================================================
    def get_IQ_integration_axis_coordinate(self, IQ_name):
        """
        returns the station coordinate values for a given IQ name
        """
        IQ_station = self.get_IQ_station(IQ_name)
        if IQ_station:
            comp_key = self.els_obj.get_IQ_attribute(IQ_name, "Component_Key")
            comp_axis = self.get_component_integration_axis(comp_key)
            if IQ_station and comp_axis:
                return self.geom_os_obj.get_station_coordinate(IQ_station, comp_axis)

        return None

    # ===========================================================================
    def get_IQ_unit(self, IQ_name):
        """
        returns the unit string for a given IQ name
        """
        if self.has_model_data and IQ_name in self.IQ_names:
            return self.els_obj.get_IQ_attribute(IQ_name, "Unit")

        return None

    # ===========================================================================
    def get_IQ_factor(self, IQ_name):
        """
        returns the conversion factor (SI -> DB) for a given IQ name
        """
        if self.has_model_data and IQ_name in self.IQ_names:
            IQ_factor = self.els_obj.get_IQ_attribute(IQ_name, "Conversion_factor")
            if IQ_factor:
                return float(IQ_factor)

        return None

    # =========================================================================
    def get_IQ_index(self, IQ_name):
        """
        returns the index of the given IQ_name in the list of all IQ names
        """
        if IQ_name in self.IQ_names:
            return self.IQ_names.index(IQ_name)

        return None

    # ===========================================================================
    def get_IQ_attribute(self, IQ_name, attr_name):
        """
        returns an attribute value for a given IQ name and attribute name
        Note: the return value is in any case a string
        """
        if self.has_model_data and IQ_name in self.IQ_names:
            return self.els_obj.get_IQ_attribute(IQ_name, attr_name)

        return None

    # =========================================================================
    def get_IQ_value(self, IQ_name, LRC_name):
        """
        returns an IQ value for a given IQ name and LRC name
        """
        if (
            self.has_LRC_values
            and IQ_name in self.IQ_names
            and LRC_name in self.LRC_index
        ):
            return self.lrtc_obj.get_IQ_value(IQ_name, LRC_name)

        return None

    # =========================================================================
    def get_IQ_def(self, IQ_name):
        """
        returns the IQ object for a given IQ name
        """
        if IQ_name in self.IQ_names:
            return self.els_obj.get_IQ_def(IQ_name)

        return None

    # =========================================================================
    def get_IQ_value_list(self, IQ_name):
        """
        returns a list with the IQ values for all LRCs for a given IQ name
        """
        if self.has_LRC_values and IQ_name in self.IQ_names:
            return self.lrtc_obj.get_IQ_column(IQ_name)

        return None

    # =========================================================================
    def has_IQ_min_max_values(self, IQ_name):
        """
        returns True if the min and max IQ values for the given IQ name exist
        """
        if IQ_name in self.IQ_names:
            if self.has_env1d_data:
                return self.lre1d_obj.has_IQ(IQ_name)

        return None

    # =========================================================================
    def get_IQ_min_max_values(self, IQ_name):
        """
        returns the min and max IQ values for all LRCs for a given IQ name
        """
        if IQ_name in self.IQ_names:
            if self.has_env1d_data:
                IQ_min_max_values = self.lre1d_obj.get_IQ_min_max_values(IQ_name)

                if IQ_min_max_values is not None:
                    return IQ_min_max_values

            if self.has_LRC_values and (
                self.calc_env_on_the_fly or IQ_name.startswith("AC")
            ):
                IQ_min_max_values = self.lrtc_obj.get_IQ_min_max_values(IQ_name)

                if IQ_min_max_values is not None:
                    return IQ_min_max_values

        return None

    # ===========================================================================
    def get_AC_param_names(self):
        """
        returns the list of AC_param IQ names
        """
        if self.has_model_data:
            return self.els_obj.get_AC_param_names()

        return None

    # =========================================================================
    def get_AC_param_data(self, IQ_name=None):
        """
        returns AC parameters data from the data set's model data
        """
        if self.has_model_data:
            return self.els_obj.get_AC_param_data(IQ_name)

        return None

    # =========================================================================
    def get_plot_data(self, comp_key=None, IQ_type=None):
        """
        returns the plot data structure from the data set's model data
        """
        if self.has_model_data:
            return self.els_obj.get_plot_data(comp_key, IQ_type)

        return None

    # =========================================================================
    def get_plot_data2(self, comp_key=None, ISO=None):
        """
        returns the plot data structure from the data set's model data
        """
        if self.has_model_data:
            return self.els_obj.get_plot_data2(comp_key, ISO)

        return None

    # ===========================================================================
    def get_comp_ISO_units(self):
        """
        returns a dict with the unit string and factor for all comp keys and ISOs
        """
        if self.has_model_data:
            return self.els_obj.get_comp_ISO_units()

        return None

    # ===========================================================================
    def has_component(self, comp_key):
        """
        checks whether comp_key is a valid component key
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            return self.geom_basic_obj.has_component(comp_key)

        return None

    # ===========================================================================
    def add_component(self, comp_key):
        """
        adds a new component to the geom_basic data
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            self.geom_basic_obj.add_component(comp_key)
            return True

        return False

    # ===========================================================================
    def set_component_data(self, comp_key, data_key, data_value):
        """
        set a component data value
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            self.geom_basic_obj.set_component_data(comp_key, data_key, data_value)
            return True

        return False

    # ===========================================================================
    def get_component_data(self, comp_key, data_key):
        """
        get a component data value
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            return self.geom_basic_obj.get_component_data(comp_key, data_key)

        return None

    # ===========================================================================
    def get_component_name(self, comp_key):
        """
        returns the component name for a given component key
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            return self.geom_basic_obj.get_component_name(comp_key)

        return None

    # ===========================================================================
    def get_component_integration_axis(self, comp_key):
        """
        returns the component integration axis for a given component key
        """
        # -----------------------------------------------------------------------

        if self.geom_basic_obj is not None:
            return self.geom_basic_obj.get_component_integration_axis(comp_key)

        return None

    # ===========================================================================
    def get_comp_stations(self, comp_key=None):
        """
        returns the lists of station objects for a given component
        """
        # -----------------------------------------------------------------------

        if self.geom_os_obj is not None:
            return self.geom_os_obj.get_comp_stations(comp_key)

        return None

    # ===========================================================================
    def get_station_def(self, station_ref):
        """
        returns a station object for a given station reference
        """
        # -----------------------------------------------------------------------

        if self.geom_os_obj is not None:
            return self.geom_os_obj.get_station_by_ref(station_ref)

        return None

    # ===========================================================================
    def has_env2d_data(self):
        """
        true if 2D envelope data available
        """
        return self.has_env2d_data

    # ===========================================================================
    def get_env2d_dict(self):
        """
        returns the dictionary of all 2D envelope combinations
        """
        if self.has_env2d_data:
            return self.lre2d_obj.get_env_dict()
        elif self.env2d_dict is not None:
            return self.env2d_dict

        return None

    # ===========================================================================
    def has_env2d(self, IQ1_name, IQ2_name):
        """
        true if the 2D envelope given by IQ1_name and IQ2_name exists
        """
        if self.has_env2d_data:
            return self.lre2d_obj.has_env(IQ1_name, IQ2_name)

        return False

    # ===========================================================================
    def get_env2d_data(self, IQ1_name, IQ2_name):
        """
        returns the data of the 2D envelope given by IQ1_name and IQ2_name
        """
        if self.has_env2d_data:
            return self.lre2d_obj.get_env_data(IQ1_name, IQ2_name)

        return None

    # ===========================================================================
    def get_CQ_dict(self):
        """
        returns the dict of 2D correlations defined in ELS
        """
        if self.has_model_data:
            return self.els_obj.get_CQ_dict()

        return None

    # ===========================================================================
    def remove_env_data(self):
        """
        removes all 1D and 2D envelope data
        """
        # move all envelope values into the lrtc object

        if self.lre1d_obj:
            self.env1d_IQ_rank_lists = self.lre1d_obj.get_all_IQ_rank_desc_lists()

        if self.lre2d_obj:
            self.env2d_dict = self.lre2d_obj.get_env_dict()

        # if  (self.lre1d_obj or self.lre2d_obj) and (not self.lrtc_obj or not self.calc_env_on_the_fly):
        if (self.lre1d_obj or self.lre2d_obj) and (
            not self.lrtc_obj or self.LRC_matrix_type == "sparse"
        ):
            if not self.lrtc_obj:
                self.lrtc_obj = LRTC_sparse_data()  # LRTC data object
                self.LRC_matrix_type = "sparse"

            if self.lre1d_obj:
                lre1d_IQ_names = self.lre1d_obj.get_IQ_names()
                for IQ_name in lre1d_IQ_names:
                    rank_desc_list = self.lre1d_obj.get_IQ_rank_desc_list(IQ_name)
                    for rank_desc in rank_desc_list:
                        (
                            IQ_value,
                            LRC_name,
                            mass_case,
                            lrtc_file,
                        ) = self.lre1d_obj.get_IQ_rank_data(IQ_name, rank_desc)
                        self.lrtc_obj.set_LRC_name(LRC_name, mass_case)
                        self.lrtc_obj.set_IQ_value(IQ_name, LRC_name, IQ_value)

            if self.lre2d_obj:
                for IQ_name1 in self.env2d_dict.keys():
                    for IQ_name2 in self.env2d_dict[IQ_name1].keys():
                        env_data = self.lre2d_obj.get_env_data(IQ_name1, IQ_name2)
                        for i_corner in range(env_data["n_corners"]):
                            LRC_name = env_data["LRC_names"][i_corner]
                            mass_case = env_data["mass_cases"][i_corner]
                            IQ_values = env_data["IQ_values"][i_corner]
                            self.lrtc_obj.set_LRC_name(LRC_name, mass_case)
                            self.lrtc_obj.set_IQ_value(IQ_name1, LRC_name, IQ_values[0])
                            self.lrtc_obj.set_IQ_value(IQ_name2, LRC_name, IQ_values[1])

            self.has_LRC_values = True
            self.calc_env_on_the_fly = True

        # -----------------------------------------------------------------------

        self.lre1d_obj = None
        self.lre2d_obj = None

        self.has_env1d_data = False
        self.has_env2d_data = False

    # ===========================================================================
    def set_calc_env_on_the_fly(self, flag):
        """
        sets the flag for envelope calculation on the fly
        """
        self.calc_env_on_the_fly = flag

    # ===========================================================================
    def transfer_IQ_corr_LRC_data(
        self, IQ_name, LRC_name, source_lrtc_obj, target_lrtc_obj
    ):
        """
        transfers the correlated LRC data for a given IQ_name from source to target lrtc object
        """
        if IQ_name.startswith("AC"):
            return

        IQ_def = self.get_IQ_def(IQ_name)
        plot_data = self.get_plot_data2(IQ_def.comp_key)

        for corr_ISO in plot_data.keys():
            if corr_ISO == IQ_def.ISO:
                continue

            corr_ISO_plot_data = plot_data[corr_ISO]

            if (
                "IQ_stations" in corr_ISO_plot_data
                and IQ_def.station in corr_ISO_plot_data["IQ_stations"]
            ):
                ind_corr_station = corr_ISO_plot_data["IQ_stations"].index(
                    IQ_def.station
                )
                corr_IQ_name = corr_ISO_plot_data["IQ_names"][ind_corr_station]
                corr_IQ_value = source_lrtc_obj.get_IQ_value(corr_IQ_name, LRC_name)

                target_lrtc_obj.set_IQ_value(corr_IQ_name, LRC_name, corr_IQ_value)

    # ===========================================================================
    def transfer_AC_param_data(self, LRC_name, source_lrtc_obj, target_lrtc_obj):
        """
        transfers the AC param data for a given LRC_name from source to target lrtc object
        """
        for AC_param_name in self.get_AC_param_names():
            AC_param_value = source_lrtc_obj.get_IQ_value(AC_param_name, LRC_name)
            target_lrtc_obj.set_IQ_value(AC_param_name, LRC_name, AC_param_value)

    # ===========================================================================
    def remove_LRC_data(self):
        """
        removes LRC data
        """
        if self.lrtc_obj:
            self.lrtc_obj = None
            self.has_LRC_values = False
            self.LRC_matrix_type = ""

    # ===========================================================================
    def write_hdf5_file(self, file_name, **args):
        """
        creates a HDF5 file
        """
        # Guess which version of HDF5 we want to write
        if "version" in args and args["version"] == 1:
            from .HDF5_Data_Set import write_hdf5_file
            return write_hdf5_file(self, file_name, **args)
        else:
            from .HDF5_Data_Set import write_hdf5_v2_file
            return write_hdf5_v2_file(self, file_name, **args)

    # ===========================================================================
    def export_ATLAS_files(self, export_path, file_name, **args):
        """
        exports the data_set into ATLAS data files

        GEOM_Basic - file_name + _GEOM_Basic.ads
        GEOM_OS    - file_name + _GEOM_OS.ads
        ELS        - file_name + _ELS.csv
        LRTC       - file_name + .lrtc
        LRE2D      - file_name + .lre1d
        LRE1D      - file_name + .lre2d
        """
        # -----------------------------------------------------------------------
        # check names

        if export_path is None or export_path == "":
            export_path = os.path.dirname(self.file_name)

        if not os.path.isdir(export_path):
            print(f"export_ATLAS_files: Invalid export path: {export_path}")
            return None

        if file_name is None or export_path == "":
            file_name = os.path.basename(self.file_name)

        # -----------------------------------------------------------------------
        # output file names

        file_path = os.path.join(export_path, file_name)

        geom_basic_file_name = file_path + "_GEOM_Basic.ads"
        geom_os_file_name = file_path + "_GEOM_OS.ads"
        els_file_name = file_path + "_ELS.csv"

        lrtc_file_name = file_path + ".lrtc"
        lre1d_file_name = file_path + ".lre1d"
        lre2d_file_name = file_path + ".lre2d"

        # -----------------------------------------------------------------------
        # control arguments

        info_data = args.get("info_data", {})

        export_flags = args.get("export_flags", {})

        # -----------------------------------------------------------------------
        # write ATLAS files

        exported_files = {}

        if self.geom_basic_obj is not None and export_flags.get("geom_basic", False):
            print(f"Export GEOM_Basic file: {geom_basic_file_name}")
            self.geom_basic_obj.write_file(geom_basic_file_name)
            exported_files["geom_basic"] = geom_basic_file_name

        if self.geom_os_obj is not None and export_flags.get("geom_os", False):
            print(f"Export GEOM_OS file: {geom_os_file_name}")
            self.geom_os_obj.write_file(geom_os_file_name)
            exported_files["geom_os"] = geom_os_file_name

        if self.els_obj is not None and export_flags.get("els", False):
            print(f"Export ELS file: {els_file_name}")
            self.els_obj.write_file(els_file_name)
            exported_files["els"] = els_file_name

        if self.lrtc_obj is not None and export_flags.get("lrtc", False):
            print(f"Export LRTC file: {lrtc_file_name}")
            self.lrtc_obj.set_info(info_data)
            self.lrtc_obj.write_file(lrtc_file_name)
            exported_files["lrtc"] = lrtc_file_name

        if export_flags.get("lre1d", False):
            if self.lre1d_obj is None:
                print("Calculate LRE1D data")
                if self.env1d_IQ_rank_lists is not None:
                    self.calc_LRE1D("LRE1D")
                else:
                    self.calc_LRE1D("all")
            if self.lre1d_obj is not None:
                print(f"Export LRE1D file: {lre1d_file_name}")
                self.lre1d_obj.set_info(info_data)
                self.lre1d_obj.write_file(lre1d_file_name)
                exported_files["lre1d"] = lre1d_file_name

        if export_flags.get("lre2d", False):
            if self.lre2d_obj is None:
                print("Calculate LRE2D data")
                if self.env2d_dict is not None:
                    self.calc_LRE2D("LRE2D")
                else:
                    self.calc_LRE2D("ELS")
            if self.lre2d_obj is not None:
                print(f"Export LRE2D file: {lre2d_file_name}")
                self.lre2d_obj.set_info(info_data)
                self.lre2d_obj.write_file(lre2d_file_name)
                exported_files["lre2d"] = lre2d_file_name

        print("Export ATLAS data files completed")

        return exported_files


#####################################################################################################
class LRC_point(CONHUL_point):
    """ """

    # ================================================================================================
    def __init__(self, x, y, LRC_name):
        CONHUL_point.__init__(self, x, y)

        self.LRC_name = LRC_name


################################################################################
