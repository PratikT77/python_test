"""
GEOM_Basic data methods
"""

################################################################################
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..data_struct_func import print_data_struct
from ..io.ads_io import structure_ads_data

logging.basicConfig(level=logging.INFO)

axis_text = ["x", "y", "z"]
axis_index = {"x": 0, "y": 1, "z": 2}

data_name_mapping = {
    "name": "Comp_Name",
    "SISO": "Super_Iso",
    "axis": "Integration_Direction",
}

################################################################################


@dataclass
class GeomBasicData(object):
    """
    GEOM_Basic data methods
    """

    file_name: str | Path = field(default=None)
    info: dict[str, Any] = field(default_factory = dict, repr = False, init = False)
    data: dict[str, Any] = field(
        default_factory = lambda: {"Components": {}}, repr = False, init = False
    )

    # ===========================================================================
    def set_ads_data(self, data_lines: list[str]):
        """
        Sets the complete data from a list of strings in .ads format.

        :param self: *object* GeomBasicData object.
        :param data_lines: *list* A list of strings, where each string is a line from an .ads file.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> ads_data_lines = ["xyz"]
        >>> geom_basic_obj.set_ads_data(ads_data_lines)
        """
        # -----------------------------------------------------------------------
        data_struct = structure_ads_data(data_lines)
        self.data = data_struct["Basic"]

    # ===========================================================================
    def get_ads_data(self) -> list[str]:
        """
        Returns the complete data as a list of strings in .ads format.

        :param self: *object* GeomBasicData object.

        :return: *list* A list of strings ready to be written to an .ads file.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> ads_data = geom_basic_obj.get_ads_data()
        """
        # -----------------------------------------------------------------------
        ads_data_list = []
        print_data_struct("Basic.", self.data, ads_data_list)
        return ads_data_list

    # ===========================================================================
    def has_component(self, comp_key: str) -> bool:
        """
        Checks whether a component key exists in the data.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The component key to check.

        :return: *bool* True if the component exists, False otherwise.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> isPresent = geom_basic_obj.has_component("FU")
        """
        # -----------------------------------------------------------------------

        return comp_key in self.data["Components"]

    # ===========================================================================
    def add_component(
        self,
        comp_key: str,
        name: Optional[str] = None,
        siso: Optional[int] = 0,
        comp_axis: str = "x",
    ):
        """
        Adds a component, creating it if it does not exist.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The unique key for the component.
        :param name: *str, optional* The name of the component. If None, defaults to the comp_key.
        :param siso: *int, optional* The SISO value for the component.
        :param comp_axis: *str* The integration direction ('x', 'y', or 'z').

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> geom_basic_obj.add_component(comp_key="WR",name="Coordinate_Systems",siso=1, comp_axis="x")
        """
        # -----------------------------------------------------------------------
        if comp_key not in self.data["Components"]:
            self.data["Components"][comp_key] = {}
            self.data["Components"][comp_key]["Data"] = {}

        comp_data = self.data["Components"][comp_key]["Data"]

        if name is None:
            name = comp_key

        comp_data["Comp_Name"] = name
        comp_data["Super_Iso"] = siso
        comp_data["Integration_Direction"] = comp_axis

    # ===========================================================================
    def set_component_data(self, comp_key: str, data_key: str, data_value):
        """
        Sets a specific data value for a given component.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component to modify.
        :param data_key: *str* The key of the data to set (e.g., 'name', 'SISO', 'iaxis').
        :param data_value: *Any* The new value to set.

        :raises KeyError: If the component key is not found in the data set.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> geom_basic_obj.set_component_data(comp_key="FU", data_key="data_key", data_value="data_value")
        """
        # -----------------------------------------------------------------------
        if not self.has_component(comp_key):
            raise KeyError(
                f"GeomBasicData.set_component_data: {comp_key}: Component key not found in data set"
            )

        if data_key == "iaxis":
            axis_str = axis_text[data_value]
            data_key = "Integration_Direction"
            data_value = axis_str

        if data_key in data_name_mapping:
            data_key = data_name_mapping[data_key]

        comp_data = self.data["Components"][comp_key]["Data"]

        comp_data[data_key] = data_value

    # ===========================================================================
    def get_component_data(self, comp_key: str, data_key: str) -> Any:
        """
        Gets a specific data value for a given component.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component to query.
        :param data_key: *str* The key of the data to retrieve (e.g., 'name', 'SISO', 'iaxis').

        :return: *Any* The requested data value.

        :raises ValueError: If the requested data_key is not found in the component data.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_data = geom_basic_obj.get_component_data(comp_key="FU", data_key="data_key")
        """
        # -----------------------------------------------------------------------

        if not self.has_component(comp_key):
            logging.warning(f"GeomBasicData.get_component_data: {comp_key}: Component key not found in data set")
            return None

        if data_key == "iaxis":
            iaxis = self.get_component_iaxis(comp_key)
            return iaxis

        if data_key in data_name_mapping:
            data_key = data_name_mapping[data_key]

        comp_data = self.data["Components"][comp_key]["Data"]

        if data_key not in comp_data:
            raise ValueError(
                f"GeomBasicData.get_component_data: Missing required data key '{data_key}' in component data: {comp_key}"
            )

        return comp_data[data_key]

    # ===========================================================================
    def get_component_data_struct(self, comp_key: str) -> dict[str, Any]:
        """
        Gets the complete data structure of a component.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component to retrieve.

        :return: *dict* A deep copy of the component's data structure.

        :raises KeyError: If the component key is not found in the data set.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_data = geom_basic_obj.get_component_data_struct(comp_key="FU")
        """
        # -----------------------------------------------------------------------
        if not self.has_component(comp_key):
            raise KeyError(
                f"GeomBasicData.get_component_data_struct: {comp_key}: Component key not found in data set"
            )

        data_struct = self.data["Components"][comp_key]

        return copy.deepcopy(data_struct)

    # ===========================================================================
    def add_component_data_struct(self, comp_key: str, data_struct: dict):
        """
        Adds the complete data struct of a component.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key for the component.
        :param data_struct: *dict* The complete data structure for the component.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> data_struct = {}
        >>> geom_basic_obj.add_component_data_struct(comp_key="WR",data_struct=data_struct)
        """
        # -----------------------------------------------------------------------

        self.data["Components"][comp_key] = copy.deepcopy(data_struct)

    # ===========================================================================
    def get_component_name(self, comp_key: str) -> str | None:
        """
        Returns the component name for a given component key.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component.

        :return: *str | None* The name of the component or none.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_name = geom_basic_obj.get_component_name(comp_key="FU")
        """
        # -----------------------------------------------------------------------

        if not self.has_component(comp_key):
            logging.warning(f"GeomBasicData.get_component_name: {comp_key}: Component key not found in data set")
            return None

        return self.data["Components"][comp_key]["Data"]["Comp_Name"]

    # ===========================================================================
    def get_component_siso(self, comp_key: str) -> int | None:
        """
        Returns the SISO value for a given component key.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component.

        :return: *int | None* The Super_Iso value of the component or None.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_siso = geom_basic_obj.get_component_siso(comp_key="FU")
        """
        # -----------------------------------------------------------------------

        if not self.has_component(comp_key):
            logging.warning(f"GeomBasicData.get_component_SISO: {comp_key}: Component key not found in data set")
            return None

        return self.data["Components"][comp_key]["Data"]["Super_Iso"]

    # ===========================================================================
    def get_component_integration_axis(self, comp_key: str) -> str | None:
        """
        Returns the component integration axis for a given component key.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component.

        :return: *str | None* The integration axis ('x', 'y', or 'z'), defaults to 'x' or none.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_int_axis = geom_basic_obj.get_component_integration_axis(comp_key="FU")
        """
        # -----------------------------------------------------------------------

        if not self.has_component(comp_key):
            logging.warning(
                f"GeomBasicData.get_component_integration_axis: {comp_key}: Component key not found in data set")
            return None

        comp_axis = self.data["Components"][comp_key]["Data"]["Integration_Direction"]

        if comp_axis is None or comp_axis == "":
            comp_axis = "x"

        return comp_axis

    # ===========================================================================
    def get_component_iaxis(self, comp_key: str) -> int | None:
        """
        Returns the index of the component integration axis.

        :param self: *object* GeomBasicData object.
        :param comp_key: *str* The key of the component.

        :return: *int | None* The numerical index of the axis (0 for x, 1 for y, 2 for z) or none value.

        :examples:
        >>> from fp_dataio.data_models.geom_basic_data import GeomBasicData
        >>> geom_basic_obj = GeomBasicData()
        >>> comp_iaxis = geom_basic_obj.get_component_iaxis(comp_key="FU")
        """
        # -----------------------------------------------------------------------
        if not self.has_component(comp_key):
            logging.warning(f"GeomBasicData.get_component_iaxis: {comp_key}: Component key not found in data set")
            return None

        axis_str = self.get_component_integration_axis(comp_key)

        return axis_index[axis_str.lower()]

    # ===========================================================================
