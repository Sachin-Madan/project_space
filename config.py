import json
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import os
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class JsonConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        encoding = self.config.get("env_file_encoding")
        file_content_json = json.loads(
            Path("config.json").read_text(encoding)
        )
        # # For debugging in VSC, please comment the above line & uncomment the below line
        # file_content_json = json.loads(
        #     Path(os.path.join("Code","config.json")).read_text(encoding)
        # )
        field_value = file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d


class Config(BaseSettings):
    
    revenue_conversion_factor : float
    volume_conversion_factor : float
    poc_id_list : list

    sku_id : str
    sku_detail : str
    poc_id : str
    year : str
    month : str
    week : str
    yearmonth : str
    yearweek : str
    revenue : str
    volume : str
    units_sold : str
    brand : str
    ptc_segment : str
    pack_size : str
    pack_qty : str
    pack_type : str
    style : str
    manufacturer : str
    price_per_ltr : str
    plano_name : str
    sku_height : str
    sku_width : str
    sku_depth : str
    intra_week_seasonality_adj_col : str
    
    total_pack_size : str
    price_per_unit : str
    cluster_parent_sku : str
    overall_vol_market_share : str
    overall_rev_market_share : str

    bool_dynamic_nesting : bool
    bool_dynamic_propensity : bool
    propensity_threshold : float
    power_of_propensity : float
    market_share_weight : float
    similarity_weight : float
    bool_pareto_dt : bool
    min_subs_needed : int
    pareto_for_subs_more_than : int
    pareto_percentile : float
    upper_limit_on_dt : float

    similarity_column_weights_dict : dict

    pack_size_bubble : float
    total_pack_size_bubble : float
    price_per_ltr_bubble : float
    price_per_unit_bubble : float
    
    remove_skus: list

    cut_off : float
    curve_column : str
    
    store_id: str
    boolMaxColInitialization : bool
    #exchange_rate: float #
    #desired_DOS_input: int #
    cut_off: float


    pareto_cut_off_opt: float
    Shelf_width_threshold: float
    #intra_week_demand_file_path: str

    iteration: int
    population: int

    plano_constraints_flag : bool
    overall_space_constraints : bool

    overall_type_constraint_precision : float
    shelf_type_constraint_precision : float
    total_planogram_space_constraint_precision : float
    
    boolHalfShelfConstraint : bool

    lstSKUsToRemove : list
    lstStackingPackType : list
    intStackingPackQty : int
    fltShelfDoorBufferHeight : float
    fltShelfDoorBufferWidth : float
    
    CLUSTER_PARENT_SKU : str
    PACK_TYPE : str
    PACK_QTY : str
    PRICE_PER_LTR : str
    REVENUE : str
    VOLUME : str
    SKU_HEIGHT : str
    SKU_WIDTH : str
    SKU_DEPTH : str
    DELISTING_NOT_ALLOWED_FLAG : str
    MIN_COLS_IF_KEPT : str
    MAX_COLS_IF_KEPT : str
    CONSTRAINT_MAPPING : str
    CONSTRAINT_ID : str
    CONSTRAINT_TYPE : str
    CONSTRAINT_SPACE_PERC : str
    CONSTRAINT_TYPE_SHELF : str
    CONSTRAINT_TYPE_OVERALL : str
    SHELF_DOOR_HEIGHT : str
    SHELF_DOOR_WIDTH : str
    SHELF_DOOR_DEPTH : str

    class Config:
        env_file = ".env"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )


# Return the settings object
# LRUCache is used to cache the settings object
config = Config()

# config.lstSKUsToRemove