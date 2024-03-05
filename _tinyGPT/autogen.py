# this is a single file representing each of the modules in the
# /runtime/autogen directory of the tinygrad project


# tinygrad/runtime/autogen/comgr.py

# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-D__HIP_PLATFORM_AMD__', '-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries = {}
_libraries['libamd_comgr.so'] = ctypes.CDLL('/opt/rocm/lib/libamd_comgr.so')
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass






# values for enumeration 'amd_comgr_status_s'
amd_comgr_status_s__enumvalues = {
    0: 'AMD_COMGR_STATUS_SUCCESS',
    1: 'AMD_COMGR_STATUS_ERROR',
    2: 'AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT',
    3: 'AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES',
}
AMD_COMGR_STATUS_SUCCESS = 0
AMD_COMGR_STATUS_ERROR = 1
AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 2
AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 3
amd_comgr_status_s = ctypes.c_uint32 # enum
amd_comgr_status_t = amd_comgr_status_s
amd_comgr_status_t__enumvalues = amd_comgr_status_s__enumvalues

# values for enumeration 'amd_comgr_language_s'
amd_comgr_language_s__enumvalues = {
    0: 'AMD_COMGR_LANGUAGE_NONE',
    1: 'AMD_COMGR_LANGUAGE_OPENCL_1_2',
    2: 'AMD_COMGR_LANGUAGE_OPENCL_2_0',
    3: 'AMD_COMGR_LANGUAGE_HC',
    4: 'AMD_COMGR_LANGUAGE_HIP',
    4: 'AMD_COMGR_LANGUAGE_LAST',
}
AMD_COMGR_LANGUAGE_NONE = 0
AMD_COMGR_LANGUAGE_OPENCL_1_2 = 1
AMD_COMGR_LANGUAGE_OPENCL_2_0 = 2
AMD_COMGR_LANGUAGE_HC = 3
AMD_COMGR_LANGUAGE_HIP = 4
AMD_COMGR_LANGUAGE_LAST = 4
amd_comgr_language_s = ctypes.c_uint32 # enum
amd_comgr_language_t = amd_comgr_language_s
amd_comgr_language_t__enumvalues = amd_comgr_language_s__enumvalues
try:
    amd_comgr_status_string = _libraries['libamd_comgr.so'].amd_comgr_status_string
    amd_comgr_status_string.restype = amd_comgr_status_t
    amd_comgr_status_string.argtypes = [amd_comgr_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    amd_comgr_get_version = _libraries['libamd_comgr.so'].amd_comgr_get_version
    amd_comgr_get_version.restype = None
    amd_comgr_get_version.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_data_kind_s'
amd_comgr_data_kind_s__enumvalues = {
    0: 'AMD_COMGR_DATA_KIND_UNDEF',
    1: 'AMD_COMGR_DATA_KIND_SOURCE',
    2: 'AMD_COMGR_DATA_KIND_INCLUDE',
    3: 'AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER',
    4: 'AMD_COMGR_DATA_KIND_DIAGNOSTIC',
    5: 'AMD_COMGR_DATA_KIND_LOG',
    6: 'AMD_COMGR_DATA_KIND_BC',
    7: 'AMD_COMGR_DATA_KIND_RELOCATABLE',
    8: 'AMD_COMGR_DATA_KIND_EXECUTABLE',
    9: 'AMD_COMGR_DATA_KIND_BYTES',
    16: 'AMD_COMGR_DATA_KIND_FATBIN',
    17: 'AMD_COMGR_DATA_KIND_AR',
    18: 'AMD_COMGR_DATA_KIND_BC_BUNDLE',
    19: 'AMD_COMGR_DATA_KIND_AR_BUNDLE',
    19: 'AMD_COMGR_DATA_KIND_LAST',
}
AMD_COMGR_DATA_KIND_UNDEF = 0
AMD_COMGR_DATA_KIND_SOURCE = 1
AMD_COMGR_DATA_KIND_INCLUDE = 2
AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 3
AMD_COMGR_DATA_KIND_DIAGNOSTIC = 4
AMD_COMGR_DATA_KIND_LOG = 5
AMD_COMGR_DATA_KIND_BC = 6
AMD_COMGR_DATA_KIND_RELOCATABLE = 7
AMD_COMGR_DATA_KIND_EXECUTABLE = 8
AMD_COMGR_DATA_KIND_BYTES = 9
AMD_COMGR_DATA_KIND_FATBIN = 16
AMD_COMGR_DATA_KIND_AR = 17
AMD_COMGR_DATA_KIND_BC_BUNDLE = 18
AMD_COMGR_DATA_KIND_AR_BUNDLE = 19
AMD_COMGR_DATA_KIND_LAST = 19
amd_comgr_data_kind_s = ctypes.c_uint32 # enum
amd_comgr_data_kind_t = amd_comgr_data_kind_s
amd_comgr_data_kind_t__enumvalues = amd_comgr_data_kind_s__enumvalues
class struct_amd_comgr_data_s(Structure):
    pass

struct_amd_comgr_data_s._pack_ = 1 # source:False
struct_amd_comgr_data_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_data_t = struct_amd_comgr_data_s
class struct_amd_comgr_data_set_s(Structure):
    pass

struct_amd_comgr_data_set_s._pack_ = 1 # source:False
struct_amd_comgr_data_set_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_data_set_t = struct_amd_comgr_data_set_s
class struct_amd_comgr_action_info_s(Structure):
    pass

struct_amd_comgr_action_info_s._pack_ = 1 # source:False
struct_amd_comgr_action_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_action_info_t = struct_amd_comgr_action_info_s
class struct_amd_comgr_metadata_node_s(Structure):
    pass

struct_amd_comgr_metadata_node_s._pack_ = 1 # source:False
struct_amd_comgr_metadata_node_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_metadata_node_t = struct_amd_comgr_metadata_node_s
class struct_amd_comgr_symbol_s(Structure):
    pass

struct_amd_comgr_symbol_s._pack_ = 1 # source:False
struct_amd_comgr_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_symbol_t = struct_amd_comgr_symbol_s
class struct_amd_comgr_disassembly_info_s(Structure):
    pass

struct_amd_comgr_disassembly_info_s._pack_ = 1 # source:False
struct_amd_comgr_disassembly_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_disassembly_info_t = struct_amd_comgr_disassembly_info_s
class struct_amd_comgr_symbolizer_info_s(Structure):
    pass

struct_amd_comgr_symbolizer_info_s._pack_ = 1 # source:False
struct_amd_comgr_symbolizer_info_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

amd_comgr_symbolizer_info_t = struct_amd_comgr_symbolizer_info_s
try:
    amd_comgr_get_isa_count = _libraries['libamd_comgr.so'].amd_comgr_get_isa_count
    amd_comgr_get_isa_count.restype = amd_comgr_status_t
    amd_comgr_get_isa_count.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    amd_comgr_get_isa_name = _libraries['libamd_comgr.so'].amd_comgr_get_isa_name
    amd_comgr_get_isa_name.restype = amd_comgr_status_t
    amd_comgr_get_isa_name.argtypes = [size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    amd_comgr_get_isa_metadata = _libraries['libamd_comgr.so'].amd_comgr_get_isa_metadata
    amd_comgr_get_isa_metadata.restype = amd_comgr_status_t
    amd_comgr_get_isa_metadata.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_create_data = _libraries['libamd_comgr.so'].amd_comgr_create_data
    amd_comgr_create_data.restype = amd_comgr_status_t
    amd_comgr_create_data.argtypes = [amd_comgr_data_kind_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_release_data = _libraries['libamd_comgr.so'].amd_comgr_release_data
    amd_comgr_release_data.restype = amd_comgr_status_t
    amd_comgr_release_data.argtypes = [amd_comgr_data_t]
except AttributeError:
    pass
try:
    amd_comgr_get_data_kind = _libraries['libamd_comgr.so'].amd_comgr_get_data_kind
    amd_comgr_get_data_kind.restype = amd_comgr_status_t
    amd_comgr_get_data_kind.argtypes = [amd_comgr_data_t, ctypes.POINTER(amd_comgr_data_kind_s)]
except AttributeError:
    pass
try:
    amd_comgr_set_data = _libraries['libamd_comgr.so'].amd_comgr_set_data
    amd_comgr_set_data.restype = amd_comgr_status_t
    amd_comgr_set_data.argtypes = [amd_comgr_data_t, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    amd_comgr_set_data_from_file_slice = _libraries['libamd_comgr.so'].amd_comgr_set_data_from_file_slice
    amd_comgr_set_data_from_file_slice.restype = amd_comgr_status_t
    amd_comgr_set_data_from_file_slice.argtypes = [amd_comgr_data_t, ctypes.c_int32, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    amd_comgr_set_data_name = _libraries['libamd_comgr.so'].amd_comgr_set_data_name
    amd_comgr_set_data_name.restype = amd_comgr_status_t
    amd_comgr_set_data_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data = _libraries['libamd_comgr.so'].amd_comgr_get_data
    amd_comgr_get_data.restype = amd_comgr_status_t
    amd_comgr_get_data.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_name = _libraries['libamd_comgr.so'].amd_comgr_get_data_name
    amd_comgr_get_data_name.restype = amd_comgr_status_t
    amd_comgr_get_data_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_isa_name = _libraries['libamd_comgr.so'].amd_comgr_get_data_isa_name
    amd_comgr_get_data_isa_name.restype = amd_comgr_status_t
    amd_comgr_get_data_isa_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_create_symbolizer_info = _libraries['libamd_comgr.so'].amd_comgr_create_symbolizer_info
    amd_comgr_create_symbolizer_info.restype = amd_comgr_status_t
    amd_comgr_create_symbolizer_info.argtypes = [amd_comgr_data_t, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)), ctypes.POINTER(struct_amd_comgr_symbolizer_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_symbolizer_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_symbolizer_info
    amd_comgr_destroy_symbolizer_info.restype = amd_comgr_status_t
    amd_comgr_destroy_symbolizer_info.argtypes = [amd_comgr_symbolizer_info_t]
except AttributeError:
    pass
try:
    amd_comgr_symbolize = _libraries['libamd_comgr.so'].amd_comgr_symbolize
    amd_comgr_symbolize.restype = amd_comgr_status_t
    amd_comgr_symbolize.argtypes = [amd_comgr_symbolizer_info_t, uint64_t, ctypes.c_bool, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_get_data_metadata = _libraries['libamd_comgr.so'].amd_comgr_get_data_metadata
    amd_comgr_get_data_metadata.restype = amd_comgr_status_t
    amd_comgr_get_data_metadata.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_metadata = _libraries['libamd_comgr.so'].amd_comgr_destroy_metadata
    amd_comgr_destroy_metadata.restype = amd_comgr_status_t
    amd_comgr_destroy_metadata.argtypes = [amd_comgr_metadata_node_t]
except AttributeError:
    pass
try:
    amd_comgr_create_data_set = _libraries['libamd_comgr.so'].amd_comgr_create_data_set
    amd_comgr_create_data_set.restype = amd_comgr_status_t
    amd_comgr_create_data_set.argtypes = [ctypes.POINTER(struct_amd_comgr_data_set_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_data_set = _libraries['libamd_comgr.so'].amd_comgr_destroy_data_set
    amd_comgr_destroy_data_set.restype = amd_comgr_status_t
    amd_comgr_destroy_data_set.argtypes = [amd_comgr_data_set_t]
except AttributeError:
    pass
try:
    amd_comgr_data_set_add = _libraries['libamd_comgr.so'].amd_comgr_data_set_add
    amd_comgr_data_set_add.restype = amd_comgr_status_t
    amd_comgr_data_set_add.argtypes = [amd_comgr_data_set_t, amd_comgr_data_t]
except AttributeError:
    pass
try:
    amd_comgr_data_set_remove = _libraries['libamd_comgr.so'].amd_comgr_data_set_remove
    amd_comgr_data_set_remove.restype = amd_comgr_status_t
    amd_comgr_data_set_remove.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t]
except AttributeError:
    pass
try:
    amd_comgr_action_data_count = _libraries['libamd_comgr.so'].amd_comgr_action_data_count
    amd_comgr_action_data_count.restype = amd_comgr_status_t
    amd_comgr_action_data_count.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_action_data_get_data = _libraries['libamd_comgr.so'].amd_comgr_action_data_get_data
    amd_comgr_action_data_get_data.restype = amd_comgr_status_t
    amd_comgr_action_data_get_data.argtypes = [amd_comgr_data_set_t, amd_comgr_data_kind_t, size_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_create_action_info = _libraries['libamd_comgr.so'].amd_comgr_create_action_info
    amd_comgr_create_action_info.restype = amd_comgr_status_t
    amd_comgr_create_action_info.argtypes = [ctypes.POINTER(struct_amd_comgr_action_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_action_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_action_info
    amd_comgr_destroy_action_info.restype = amd_comgr_status_t
    amd_comgr_destroy_action_info.argtypes = [amd_comgr_action_info_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_isa_name = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_isa_name
    amd_comgr_action_info_set_isa_name.restype = amd_comgr_status_t
    amd_comgr_action_info_set_isa_name.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_isa_name = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_isa_name
    amd_comgr_action_info_get_isa_name.restype = amd_comgr_status_t
    amd_comgr_action_info_get_isa_name.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_language = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_language
    amd_comgr_action_info_set_language.restype = amd_comgr_status_t
    amd_comgr_action_info_set_language.argtypes = [amd_comgr_action_info_t, amd_comgr_language_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_language = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_language
    amd_comgr_action_info_get_language.restype = amd_comgr_status_t
    amd_comgr_action_info_get_language.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(amd_comgr_language_s)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_options = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_options
    amd_comgr_action_info_set_options.restype = amd_comgr_status_t
    amd_comgr_action_info_set_options.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_options = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_options
    amd_comgr_action_info_get_options.restype = amd_comgr_status_t
    amd_comgr_action_info_get_options.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_option_list = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_option_list
    amd_comgr_action_info_set_option_list.restype = amd_comgr_status_t
    amd_comgr_action_info_set_option_list.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char) * 0, size_t]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_option_list_count = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_option_list_count
    amd_comgr_action_info_get_option_list_count.restype = amd_comgr_status_t
    amd_comgr_action_info_get_option_list_count.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_option_list_item = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_option_list_item
    amd_comgr_action_info_get_option_list_item.restype = amd_comgr_status_t
    amd_comgr_action_info_get_option_list_item.argtypes = [amd_comgr_action_info_t, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_working_directory_path = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_working_directory_path
    amd_comgr_action_info_set_working_directory_path.restype = amd_comgr_status_t
    amd_comgr_action_info_set_working_directory_path.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_working_directory_path = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_working_directory_path
    amd_comgr_action_info_get_working_directory_path.restype = amd_comgr_status_t
    amd_comgr_action_info_get_working_directory_path.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_action_info_set_logging = _libraries['libamd_comgr.so'].amd_comgr_action_info_set_logging
    amd_comgr_action_info_set_logging.restype = amd_comgr_status_t
    amd_comgr_action_info_set_logging.argtypes = [amd_comgr_action_info_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    amd_comgr_action_info_get_logging = _libraries['libamd_comgr.so'].amd_comgr_action_info_get_logging
    amd_comgr_action_info_get_logging.restype = amd_comgr_status_t
    amd_comgr_action_info_get_logging.argtypes = [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_action_kind_s'
amd_comgr_action_kind_s__enumvalues = {
    0: 'AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR',
    1: 'AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS',
    2: 'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC',
    3: 'AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES',
    4: 'AMD_COMGR_ACTION_LINK_BC_TO_BC',
    5: 'AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC',
    6: 'AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE',
    7: 'AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY',
    8: 'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE',
    9: 'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE',
    10: 'AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE',
    11: 'AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE',
    12: 'AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE',
    13: 'AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE',
    14: 'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN',
    15: 'AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC',
    15: 'AMD_COMGR_ACTION_LAST',
}
AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = 0
AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = 1
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 2
AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES = 3
AMD_COMGR_ACTION_LINK_BC_TO_BC = 4
AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC = 5
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 6
AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = 7
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = 8
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 9
AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = 10
AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = 11
AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = 12
AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = 13
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN = 14
AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 15
AMD_COMGR_ACTION_LAST = 15
amd_comgr_action_kind_s = ctypes.c_uint32 # enum
amd_comgr_action_kind_t = amd_comgr_action_kind_s
amd_comgr_action_kind_t__enumvalues = amd_comgr_action_kind_s__enumvalues
try:
    amd_comgr_do_action = _libraries['libamd_comgr.so'].amd_comgr_do_action
    amd_comgr_do_action.restype = amd_comgr_status_t
    amd_comgr_do_action.argtypes = [amd_comgr_action_kind_t, amd_comgr_action_info_t, amd_comgr_data_set_t, amd_comgr_data_set_t]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_metadata_kind_s'
amd_comgr_metadata_kind_s__enumvalues = {
    0: 'AMD_COMGR_METADATA_KIND_NULL',
    1: 'AMD_COMGR_METADATA_KIND_STRING',
    2: 'AMD_COMGR_METADATA_KIND_MAP',
    3: 'AMD_COMGR_METADATA_KIND_LIST',
    3: 'AMD_COMGR_METADATA_KIND_LAST',
}
AMD_COMGR_METADATA_KIND_NULL = 0
AMD_COMGR_METADATA_KIND_STRING = 1
AMD_COMGR_METADATA_KIND_MAP = 2
AMD_COMGR_METADATA_KIND_LIST = 3
AMD_COMGR_METADATA_KIND_LAST = 3
amd_comgr_metadata_kind_s = ctypes.c_uint32 # enum
amd_comgr_metadata_kind_t = amd_comgr_metadata_kind_s
amd_comgr_metadata_kind_t__enumvalues = amd_comgr_metadata_kind_s__enumvalues
try:
    amd_comgr_get_metadata_kind = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_kind
    amd_comgr_get_metadata_kind.restype = amd_comgr_status_t
    amd_comgr_get_metadata_kind.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(amd_comgr_metadata_kind_s)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_string = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_string
    amd_comgr_get_metadata_string.restype = amd_comgr_status_t
    amd_comgr_get_metadata_string.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_map_size = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_map_size
    amd_comgr_get_metadata_map_size.restype = amd_comgr_status_t
    amd_comgr_get_metadata_map_size.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_iterate_map_metadata = _libraries['libamd_comgr.so'].amd_comgr_iterate_map_metadata
    amd_comgr_iterate_map_metadata.restype = amd_comgr_status_t
    amd_comgr_iterate_map_metadata.argtypes = [amd_comgr_metadata_node_t, ctypes.CFUNCTYPE(amd_comgr_status_s, struct_amd_comgr_metadata_node_s, struct_amd_comgr_metadata_node_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_metadata_lookup = _libraries['libamd_comgr.so'].amd_comgr_metadata_lookup
    amd_comgr_metadata_lookup.restype = amd_comgr_status_t
    amd_comgr_metadata_lookup.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_get_metadata_list_size = _libraries['libamd_comgr.so'].amd_comgr_get_metadata_list_size
    amd_comgr_get_metadata_list_size.restype = amd_comgr_status_t
    amd_comgr_get_metadata_list_size.argtypes = [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_index_list_metadata = _libraries['libamd_comgr.so'].amd_comgr_index_list_metadata
    amd_comgr_index_list_metadata.restype = amd_comgr_status_t
    amd_comgr_index_list_metadata.argtypes = [amd_comgr_metadata_node_t, size_t, ctypes.POINTER(struct_amd_comgr_metadata_node_s)]
except AttributeError:
    pass
try:
    amd_comgr_iterate_symbols = _libraries['libamd_comgr.so'].amd_comgr_iterate_symbols
    amd_comgr_iterate_symbols.restype = amd_comgr_status_t
    amd_comgr_iterate_symbols.argtypes = [amd_comgr_data_t, ctypes.CFUNCTYPE(amd_comgr_status_s, struct_amd_comgr_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_symbol_lookup = _libraries['libamd_comgr.so'].amd_comgr_symbol_lookup
    amd_comgr_symbol_lookup.restype = amd_comgr_status_t
    amd_comgr_symbol_lookup.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_amd_comgr_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'amd_comgr_symbol_type_s'
amd_comgr_symbol_type_s__enumvalues = {
    -1: 'AMD_COMGR_SYMBOL_TYPE_UNKNOWN',
    0: 'AMD_COMGR_SYMBOL_TYPE_NOTYPE',
    1: 'AMD_COMGR_SYMBOL_TYPE_OBJECT',
    2: 'AMD_COMGR_SYMBOL_TYPE_FUNC',
    3: 'AMD_COMGR_SYMBOL_TYPE_SECTION',
    4: 'AMD_COMGR_SYMBOL_TYPE_FILE',
    5: 'AMD_COMGR_SYMBOL_TYPE_COMMON',
    10: 'AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL',
}
AMD_COMGR_SYMBOL_TYPE_UNKNOWN = -1
AMD_COMGR_SYMBOL_TYPE_NOTYPE = 0
AMD_COMGR_SYMBOL_TYPE_OBJECT = 1
AMD_COMGR_SYMBOL_TYPE_FUNC = 2
AMD_COMGR_SYMBOL_TYPE_SECTION = 3
AMD_COMGR_SYMBOL_TYPE_FILE = 4
AMD_COMGR_SYMBOL_TYPE_COMMON = 5
AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL = 10
amd_comgr_symbol_type_s = ctypes.c_int32 # enum
amd_comgr_symbol_type_t = amd_comgr_symbol_type_s
amd_comgr_symbol_type_t__enumvalues = amd_comgr_symbol_type_s__enumvalues

# values for enumeration 'amd_comgr_symbol_info_s'
amd_comgr_symbol_info_s__enumvalues = {
    0: 'AMD_COMGR_SYMBOL_INFO_NAME_LENGTH',
    1: 'AMD_COMGR_SYMBOL_INFO_NAME',
    2: 'AMD_COMGR_SYMBOL_INFO_TYPE',
    3: 'AMD_COMGR_SYMBOL_INFO_SIZE',
    4: 'AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED',
    5: 'AMD_COMGR_SYMBOL_INFO_VALUE',
    5: 'AMD_COMGR_SYMBOL_INFO_LAST',
}
AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = 0
AMD_COMGR_SYMBOL_INFO_NAME = 1
AMD_COMGR_SYMBOL_INFO_TYPE = 2
AMD_COMGR_SYMBOL_INFO_SIZE = 3
AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED = 4
AMD_COMGR_SYMBOL_INFO_VALUE = 5
AMD_COMGR_SYMBOL_INFO_LAST = 5
amd_comgr_symbol_info_s = ctypes.c_uint32 # enum
amd_comgr_symbol_info_t = amd_comgr_symbol_info_s
amd_comgr_symbol_info_t__enumvalues = amd_comgr_symbol_info_s__enumvalues
try:
    amd_comgr_symbol_get_info = _libraries['libamd_comgr.so'].amd_comgr_symbol_get_info
    amd_comgr_symbol_get_info.restype = amd_comgr_status_t
    amd_comgr_symbol_get_info.argtypes = [amd_comgr_symbol_t, amd_comgr_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    amd_comgr_create_disassembly_info = _libraries['libamd_comgr.so'].amd_comgr_create_disassembly_info
    amd_comgr_create_disassembly_info.restype = amd_comgr_status_t
    amd_comgr_create_disassembly_info.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)), ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(struct_amd_comgr_disassembly_info_s)]
except AttributeError:
    pass
try:
    amd_comgr_destroy_disassembly_info = _libraries['libamd_comgr.so'].amd_comgr_destroy_disassembly_info
    amd_comgr_destroy_disassembly_info.restype = amd_comgr_status_t
    amd_comgr_destroy_disassembly_info.argtypes = [amd_comgr_disassembly_info_t]
except AttributeError:
    pass
try:
    amd_comgr_disassemble_instruction = _libraries['libamd_comgr.so'].amd_comgr_disassemble_instruction
    amd_comgr_disassemble_instruction.restype = amd_comgr_status_t
    amd_comgr_disassemble_instruction.argtypes = [amd_comgr_disassembly_info_t, uint64_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_demangle_symbol_name = _libraries['libamd_comgr.so'].amd_comgr_demangle_symbol_name
    amd_comgr_demangle_symbol_name.restype = amd_comgr_status_t
    amd_comgr_demangle_symbol_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_amd_comgr_data_s)]
except AttributeError:
    pass
try:
    amd_comgr_populate_mangled_names = _libraries['libamd_comgr.so'].amd_comgr_populate_mangled_names
    amd_comgr_populate_mangled_names.restype = amd_comgr_status_t
    amd_comgr_populate_mangled_names.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_get_mangled_name = _libraries['libamd_comgr.so'].amd_comgr_get_mangled_name
    amd_comgr_get_mangled_name.restype = amd_comgr_status_t
    amd_comgr_get_mangled_name.argtypes = [amd_comgr_data_t, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    amd_comgr_populate_name_expression_map = _libraries['libamd_comgr.so'].amd_comgr_populate_name_expression_map
    amd_comgr_populate_name_expression_map.restype = amd_comgr_status_t
    amd_comgr_populate_name_expression_map.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    amd_comgr_map_name_expression_to_symbol_name = _libraries['libamd_comgr.so'].amd_comgr_map_name_expression_to_symbol_name
    amd_comgr_map_name_expression_to_symbol_name.restype = amd_comgr_status_t
    amd_comgr_map_name_expression_to_symbol_name.argtypes = [amd_comgr_data_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct_code_object_info_s(Structure):
    pass

struct_code_object_info_s._pack_ = 1 # source:False
struct_code_object_info_s._fields_ = [
    ('isa', ctypes.POINTER(ctypes.c_char)),
    ('size', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

amd_comgr_code_object_info_t = struct_code_object_info_s
try:
    amd_comgr_lookup_code_object = _libraries['libamd_comgr.so'].amd_comgr_lookup_code_object
    amd_comgr_lookup_code_object.restype = amd_comgr_status_t
    amd_comgr_lookup_code_object.argtypes = [amd_comgr_data_t, ctypes.POINTER(struct_code_object_info_s), size_t]
except AttributeError:
    pass
__all__ = \
    ['AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES',
    'AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS',
    'AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY',
    'AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN',
    'AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC',
    'AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE',
    'AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE',
    'AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE',
    'AMD_COMGR_ACTION_LAST', 'AMD_COMGR_ACTION_LINK_BC_TO_BC',
    'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE',
    'AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE',
    'AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC',
    'AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR',
    'AMD_COMGR_DATA_KIND_AR', 'AMD_COMGR_DATA_KIND_AR_BUNDLE',
    'AMD_COMGR_DATA_KIND_BC', 'AMD_COMGR_DATA_KIND_BC_BUNDLE',
    'AMD_COMGR_DATA_KIND_BYTES', 'AMD_COMGR_DATA_KIND_DIAGNOSTIC',
    'AMD_COMGR_DATA_KIND_EXECUTABLE', 'AMD_COMGR_DATA_KIND_FATBIN',
    'AMD_COMGR_DATA_KIND_INCLUDE', 'AMD_COMGR_DATA_KIND_LAST',
    'AMD_COMGR_DATA_KIND_LOG',
    'AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER',
    'AMD_COMGR_DATA_KIND_RELOCATABLE', 'AMD_COMGR_DATA_KIND_SOURCE',
    'AMD_COMGR_DATA_KIND_UNDEF', 'AMD_COMGR_LANGUAGE_HC',
    'AMD_COMGR_LANGUAGE_HIP', 'AMD_COMGR_LANGUAGE_LAST',
    'AMD_COMGR_LANGUAGE_NONE', 'AMD_COMGR_LANGUAGE_OPENCL_1_2',
    'AMD_COMGR_LANGUAGE_OPENCL_2_0', 'AMD_COMGR_METADATA_KIND_LAST',
    'AMD_COMGR_METADATA_KIND_LIST', 'AMD_COMGR_METADATA_KIND_MAP',
    'AMD_COMGR_METADATA_KIND_NULL', 'AMD_COMGR_METADATA_KIND_STRING',
    'AMD_COMGR_STATUS_ERROR',
    'AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT',
    'AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES',
    'AMD_COMGR_STATUS_SUCCESS', 'AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED',
    'AMD_COMGR_SYMBOL_INFO_LAST', 'AMD_COMGR_SYMBOL_INFO_NAME',
    'AMD_COMGR_SYMBOL_INFO_NAME_LENGTH', 'AMD_COMGR_SYMBOL_INFO_SIZE',
    'AMD_COMGR_SYMBOL_INFO_TYPE', 'AMD_COMGR_SYMBOL_INFO_VALUE',
    'AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL',
    'AMD_COMGR_SYMBOL_TYPE_COMMON', 'AMD_COMGR_SYMBOL_TYPE_FILE',
    'AMD_COMGR_SYMBOL_TYPE_FUNC', 'AMD_COMGR_SYMBOL_TYPE_NOTYPE',
    'AMD_COMGR_SYMBOL_TYPE_OBJECT', 'AMD_COMGR_SYMBOL_TYPE_SECTION',
    'AMD_COMGR_SYMBOL_TYPE_UNKNOWN', 'amd_comgr_action_data_count',
    'amd_comgr_action_data_get_data',
    'amd_comgr_action_info_get_isa_name',
    'amd_comgr_action_info_get_language',
    'amd_comgr_action_info_get_logging',
    'amd_comgr_action_info_get_option_list_count',
    'amd_comgr_action_info_get_option_list_item',
    'amd_comgr_action_info_get_options',
    'amd_comgr_action_info_get_working_directory_path',
    'amd_comgr_action_info_set_isa_name',
    'amd_comgr_action_info_set_language',
    'amd_comgr_action_info_set_logging',
    'amd_comgr_action_info_set_option_list',
    'amd_comgr_action_info_set_options',
    'amd_comgr_action_info_set_working_directory_path',
    'amd_comgr_action_info_t', 'amd_comgr_action_kind_s',
    'amd_comgr_action_kind_t', 'amd_comgr_action_kind_t__enumvalues',
    'amd_comgr_code_object_info_t', 'amd_comgr_create_action_info',
    'amd_comgr_create_data', 'amd_comgr_create_data_set',
    'amd_comgr_create_disassembly_info',
    'amd_comgr_create_symbolizer_info', 'amd_comgr_data_kind_s',
    'amd_comgr_data_kind_t', 'amd_comgr_data_kind_t__enumvalues',
    'amd_comgr_data_set_add', 'amd_comgr_data_set_remove',
    'amd_comgr_data_set_t', 'amd_comgr_data_t',
    'amd_comgr_demangle_symbol_name', 'amd_comgr_destroy_action_info',
    'amd_comgr_destroy_data_set',
    'amd_comgr_destroy_disassembly_info',
    'amd_comgr_destroy_metadata', 'amd_comgr_destroy_symbolizer_info',
    'amd_comgr_disassemble_instruction',
    'amd_comgr_disassembly_info_t', 'amd_comgr_do_action',
    'amd_comgr_get_data', 'amd_comgr_get_data_isa_name',
    'amd_comgr_get_data_kind', 'amd_comgr_get_data_metadata',
    'amd_comgr_get_data_name', 'amd_comgr_get_isa_count',
    'amd_comgr_get_isa_metadata', 'amd_comgr_get_isa_name',
    'amd_comgr_get_mangled_name', 'amd_comgr_get_metadata_kind',
    'amd_comgr_get_metadata_list_size',
    'amd_comgr_get_metadata_map_size',
    'amd_comgr_get_metadata_string', 'amd_comgr_get_version',
    'amd_comgr_index_list_metadata', 'amd_comgr_iterate_map_metadata',
    'amd_comgr_iterate_symbols', 'amd_comgr_language_s',
    'amd_comgr_language_t', 'amd_comgr_language_t__enumvalues',
    'amd_comgr_lookup_code_object',
    'amd_comgr_map_name_expression_to_symbol_name',
    'amd_comgr_metadata_kind_s', 'amd_comgr_metadata_kind_t',
    'amd_comgr_metadata_kind_t__enumvalues',
    'amd_comgr_metadata_lookup', 'amd_comgr_metadata_node_t',
    'amd_comgr_populate_mangled_names',
    'amd_comgr_populate_name_expression_map',
    'amd_comgr_release_data', 'amd_comgr_set_data',
    'amd_comgr_set_data_from_file_slice', 'amd_comgr_set_data_name',
    'amd_comgr_status_s', 'amd_comgr_status_string',
    'amd_comgr_status_t', 'amd_comgr_status_t__enumvalues',
    'amd_comgr_symbol_get_info', 'amd_comgr_symbol_info_s',
    'amd_comgr_symbol_info_t', 'amd_comgr_symbol_info_t__enumvalues',
    'amd_comgr_symbol_lookup', 'amd_comgr_symbol_t',
    'amd_comgr_symbol_type_s', 'amd_comgr_symbol_type_t',
    'amd_comgr_symbol_type_t__enumvalues', 'amd_comgr_symbolize',
    'amd_comgr_symbolizer_info_t', 'size_t',
    'struct_amd_comgr_action_info_s', 'struct_amd_comgr_data_s',
    'struct_amd_comgr_data_set_s',
    'struct_amd_comgr_disassembly_info_s',
    'struct_amd_comgr_metadata_node_s', 'struct_amd_comgr_symbol_s',
    'struct_amd_comgr_symbolizer_info_s', 'struct_code_object_info_s',
    'uint64_t']


# tinygrad/runtime/autogen/cuda.py

# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries = {}
_libraries['libcuda.so'] = ctypes.CDLL(ctypes.util.find_library('cuda'))
_libraries['libnvrtc.so'] = ctypes.CDLL(ctypes.util.find_library('nvrtc'))


cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64
CUdeviceptr_v2 = ctypes.c_uint64
CUdeviceptr = ctypes.c_uint64
CUdevice_v1 = ctypes.c_int32
CUdevice = ctypes.c_int32
class struct_CUctx_st(Structure):
    pass

CUcontext = ctypes.POINTER(struct_CUctx_st)
class struct_CUmod_st(Structure):
    pass

CUmodule = ctypes.POINTER(struct_CUmod_st)
class struct_CUfunc_st(Structure):
    pass

CUfunction = ctypes.POINTER(struct_CUfunc_st)
class struct_CUarray_st(Structure):
    pass

CUarray = ctypes.POINTER(struct_CUarray_st)
class struct_CUmipmappedArray_st(Structure):
    pass

CUmipmappedArray = ctypes.POINTER(struct_CUmipmappedArray_st)
class struct_CUtexref_st(Structure):
    pass

CUtexref = ctypes.POINTER(struct_CUtexref_st)
class struct_CUsurfref_st(Structure):
    pass

CUsurfref = ctypes.POINTER(struct_CUsurfref_st)
class struct_CUevent_st(Structure):
    pass

CUevent = ctypes.POINTER(struct_CUevent_st)
class struct_CUstream_st(Structure):
    pass

CUstream = ctypes.POINTER(struct_CUstream_st)
class struct_CUgraphicsResource_st(Structure):
    pass

CUgraphicsResource = ctypes.POINTER(struct_CUgraphicsResource_st)
CUtexObject_v1 = ctypes.c_uint64
CUtexObject = ctypes.c_uint64
CUsurfObject_v1 = ctypes.c_uint64
CUsurfObject = ctypes.c_uint64
class struct_CUextMemory_st(Structure):
    pass

CUexternalMemory = ctypes.POINTER(struct_CUextMemory_st)
class struct_CUextSemaphore_st(Structure):
    pass

CUexternalSemaphore = ctypes.POINTER(struct_CUextSemaphore_st)
class struct_CUgraph_st(Structure):
    pass

CUgraph = ctypes.POINTER(struct_CUgraph_st)
class struct_CUgraphNode_st(Structure):
    pass

CUgraphNode = ctypes.POINTER(struct_CUgraphNode_st)
class struct_CUgraphExec_st(Structure):
    pass

CUgraphExec = ctypes.POINTER(struct_CUgraphExec_st)
class struct_CUmemPoolHandle_st(Structure):
    pass

CUmemoryPool = ctypes.POINTER(struct_CUmemPoolHandle_st)
class struct_CUuserObject_st(Structure):
    pass

CUuserObject = ctypes.POINTER(struct_CUuserObject_st)
class struct_CUuuid_st(Structure):
    pass

struct_CUuuid_st._pack_ = 1 # source:False
struct_CUuuid_st._fields_ = [
    ('bytes', ctypes.c_char * 16),
]

CUuuid = struct_CUuuid_st
class struct_CUipcEventHandle_st(Structure):
    pass

struct_CUipcEventHandle_st._pack_ = 1 # source:False
struct_CUipcEventHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

CUipcEventHandle_v1 = struct_CUipcEventHandle_st
CUipcEventHandle = struct_CUipcEventHandle_st
class struct_CUipcMemHandle_st(Structure):
    pass

struct_CUipcMemHandle_st._pack_ = 1 # source:False
struct_CUipcMemHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

CUipcMemHandle_v1 = struct_CUipcMemHandle_st
CUipcMemHandle = struct_CUipcMemHandle_st

# values for enumeration 'CUipcMem_flags_enum'
CUipcMem_flags_enum__enumvalues = {
    1: 'CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS',
}
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
CUipcMem_flags_enum = ctypes.c_uint32 # enum
CUipcMem_flags = CUipcMem_flags_enum
CUipcMem_flags__enumvalues = CUipcMem_flags_enum__enumvalues

# values for enumeration 'CUmemAttach_flags_enum'
CUmemAttach_flags_enum__enumvalues = {
    1: 'CU_MEM_ATTACH_GLOBAL',
    2: 'CU_MEM_ATTACH_HOST',
    4: 'CU_MEM_ATTACH_SINGLE',
}
CU_MEM_ATTACH_GLOBAL = 1
CU_MEM_ATTACH_HOST = 2
CU_MEM_ATTACH_SINGLE = 4
CUmemAttach_flags_enum = ctypes.c_uint32 # enum
CUmemAttach_flags = CUmemAttach_flags_enum
CUmemAttach_flags__enumvalues = CUmemAttach_flags_enum__enumvalues

# values for enumeration 'CUctx_flags_enum'
CUctx_flags_enum__enumvalues = {
    0: 'CU_CTX_SCHED_AUTO',
    1: 'CU_CTX_SCHED_SPIN',
    2: 'CU_CTX_SCHED_YIELD',
    4: 'CU_CTX_SCHED_BLOCKING_SYNC',
    4: 'CU_CTX_BLOCKING_SYNC',
    7: 'CU_CTX_SCHED_MASK',
    8: 'CU_CTX_MAP_HOST',
    16: 'CU_CTX_LMEM_RESIZE_TO_MAX',
    31: 'CU_CTX_FLAGS_MASK',
}
CU_CTX_SCHED_AUTO = 0
CU_CTX_SCHED_SPIN = 1
CU_CTX_SCHED_YIELD = 2
CU_CTX_SCHED_BLOCKING_SYNC = 4
CU_CTX_BLOCKING_SYNC = 4
CU_CTX_SCHED_MASK = 7
CU_CTX_MAP_HOST = 8
CU_CTX_LMEM_RESIZE_TO_MAX = 16
CU_CTX_FLAGS_MASK = 31
CUctx_flags_enum = ctypes.c_uint32 # enum
CUctx_flags = CUctx_flags_enum
CUctx_flags__enumvalues = CUctx_flags_enum__enumvalues

# values for enumeration 'CUstream_flags_enum'
CUstream_flags_enum__enumvalues = {
    0: 'CU_STREAM_DEFAULT',
    1: 'CU_STREAM_NON_BLOCKING',
}
CU_STREAM_DEFAULT = 0
CU_STREAM_NON_BLOCKING = 1
CUstream_flags_enum = ctypes.c_uint32 # enum
CUstream_flags = CUstream_flags_enum
CUstream_flags__enumvalues = CUstream_flags_enum__enumvalues

# values for enumeration 'CUevent_flags_enum'
CUevent_flags_enum__enumvalues = {
    0: 'CU_EVENT_DEFAULT',
    1: 'CU_EVENT_BLOCKING_SYNC',
    2: 'CU_EVENT_DISABLE_TIMING',
    4: 'CU_EVENT_INTERPROCESS',
}
CU_EVENT_DEFAULT = 0
CU_EVENT_BLOCKING_SYNC = 1
CU_EVENT_DISABLE_TIMING = 2
CU_EVENT_INTERPROCESS = 4
CUevent_flags_enum = ctypes.c_uint32 # enum
CUevent_flags = CUevent_flags_enum
CUevent_flags__enumvalues = CUevent_flags_enum__enumvalues

# values for enumeration 'CUevent_record_flags_enum'
CUevent_record_flags_enum__enumvalues = {
    0: 'CU_EVENT_RECORD_DEFAULT',
    1: 'CU_EVENT_RECORD_EXTERNAL',
}
CU_EVENT_RECORD_DEFAULT = 0
CU_EVENT_RECORD_EXTERNAL = 1
CUevent_record_flags_enum = ctypes.c_uint32 # enum
CUevent_record_flags = CUevent_record_flags_enum
CUevent_record_flags__enumvalues = CUevent_record_flags_enum__enumvalues

# values for enumeration 'CUevent_wait_flags_enum'
CUevent_wait_flags_enum__enumvalues = {
    0: 'CU_EVENT_WAIT_DEFAULT',
    1: 'CU_EVENT_WAIT_EXTERNAL',
}
CU_EVENT_WAIT_DEFAULT = 0
CU_EVENT_WAIT_EXTERNAL = 1
CUevent_wait_flags_enum = ctypes.c_uint32 # enum
CUevent_wait_flags = CUevent_wait_flags_enum
CUevent_wait_flags__enumvalues = CUevent_wait_flags_enum__enumvalues

# values for enumeration 'CUstreamWaitValue_flags_enum'
CUstreamWaitValue_flags_enum__enumvalues = {
    0: 'CU_STREAM_WAIT_VALUE_GEQ',
    1: 'CU_STREAM_WAIT_VALUE_EQ',
    2: 'CU_STREAM_WAIT_VALUE_AND',
    3: 'CU_STREAM_WAIT_VALUE_NOR',
    1073741824: 'CU_STREAM_WAIT_VALUE_FLUSH',
}
CU_STREAM_WAIT_VALUE_GEQ = 0
CU_STREAM_WAIT_VALUE_EQ = 1
CU_STREAM_WAIT_VALUE_AND = 2
CU_STREAM_WAIT_VALUE_NOR = 3
CU_STREAM_WAIT_VALUE_FLUSH = 1073741824
CUstreamWaitValue_flags_enum = ctypes.c_uint32 # enum
CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum
CUstreamWaitValue_flags__enumvalues = CUstreamWaitValue_flags_enum__enumvalues

# values for enumeration 'CUstreamWriteValue_flags_enum'
CUstreamWriteValue_flags_enum__enumvalues = {
    0: 'CU_STREAM_WRITE_VALUE_DEFAULT',
    1: 'CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER',
}
CU_STREAM_WRITE_VALUE_DEFAULT = 0
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1
CUstreamWriteValue_flags_enum = ctypes.c_uint32 # enum
CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum
CUstreamWriteValue_flags__enumvalues = CUstreamWriteValue_flags_enum__enumvalues

# values for enumeration 'CUstreamBatchMemOpType_enum'
CUstreamBatchMemOpType_enum__enumvalues = {
    1: 'CU_STREAM_MEM_OP_WAIT_VALUE_32',
    2: 'CU_STREAM_MEM_OP_WRITE_VALUE_32',
    4: 'CU_STREAM_MEM_OP_WAIT_VALUE_64',
    5: 'CU_STREAM_MEM_OP_WRITE_VALUE_64',
    3: 'CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES',
}
CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4
CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
CUstreamBatchMemOpType_enum = ctypes.c_uint32 # enum
CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum
CUstreamBatchMemOpType__enumvalues = CUstreamBatchMemOpType_enum__enumvalues
class union_CUstreamBatchMemOpParams_union(Union):
    pass

class struct_CUstreamMemOpWaitValueParams_st(Structure):
    pass

class union_CUstreamMemOpWaitValueParams_st_0(Union):
    pass

union_CUstreamMemOpWaitValueParams_st_0._pack_ = 1 # source:False
union_CUstreamMemOpWaitValueParams_st_0._fields_ = [
    ('value', ctypes.c_uint32),
    ('value64', ctypes.c_uint64),
]

struct_CUstreamMemOpWaitValueParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpWaitValueParams_st._anonymous_ = ('_0',)
struct_CUstreamMemOpWaitValueParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('address', ctypes.c_uint64),
    ('_0', union_CUstreamMemOpWaitValueParams_st_0),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('alias', ctypes.c_uint64),
]

class struct_CUstreamMemOpWriteValueParams_st(Structure):
    pass

class union_CUstreamMemOpWriteValueParams_st_0(Union):
    pass

union_CUstreamMemOpWriteValueParams_st_0._pack_ = 1 # source:False
union_CUstreamMemOpWriteValueParams_st_0._fields_ = [
    ('value', ctypes.c_uint32),
    ('value64', ctypes.c_uint64),
]

struct_CUstreamMemOpWriteValueParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpWriteValueParams_st._anonymous_ = ('_0',)
struct_CUstreamMemOpWriteValueParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('address', ctypes.c_uint64),
    ('_0', union_CUstreamMemOpWriteValueParams_st_0),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('alias', ctypes.c_uint64),
]

class struct_CUstreamMemOpFlushRemoteWritesParams_st(Structure):
    pass

struct_CUstreamMemOpFlushRemoteWritesParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpFlushRemoteWritesParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('flags', ctypes.c_uint32),
]

union_CUstreamBatchMemOpParams_union._pack_ = 1 # source:False
union_CUstreamBatchMemOpParams_union._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('waitValue', struct_CUstreamMemOpWaitValueParams_st),
    ('writeValue', struct_CUstreamMemOpWriteValueParams_st),
    ('flushRemoteWrites', struct_CUstreamMemOpFlushRemoteWritesParams_st),
    ('pad', ctypes.c_uint64 * 6),
]

CUstreamBatchMemOpParams_v1 = union_CUstreamBatchMemOpParams_union
CUstreamBatchMemOpParams = union_CUstreamBatchMemOpParams_union

# values for enumeration 'CUoccupancy_flags_enum'
CUoccupancy_flags_enum__enumvalues = {
    0: 'CU_OCCUPANCY_DEFAULT',
    1: 'CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE',
}
CU_OCCUPANCY_DEFAULT = 0
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1
CUoccupancy_flags_enum = ctypes.c_uint32 # enum
CUoccupancy_flags = CUoccupancy_flags_enum
CUoccupancy_flags__enumvalues = CUoccupancy_flags_enum__enumvalues

# values for enumeration 'CUstreamUpdateCaptureDependencies_flags_enum'
CUstreamUpdateCaptureDependencies_flags_enum__enumvalues = {
    0: 'CU_STREAM_ADD_CAPTURE_DEPENDENCIES',
    1: 'CU_STREAM_SET_CAPTURE_DEPENDENCIES',
}
CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0
CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1
CUstreamUpdateCaptureDependencies_flags_enum = ctypes.c_uint32 # enum
CUstreamUpdateCaptureDependencies_flags = CUstreamUpdateCaptureDependencies_flags_enum
CUstreamUpdateCaptureDependencies_flags__enumvalues = CUstreamUpdateCaptureDependencies_flags_enum__enumvalues

# values for enumeration 'CUarray_format_enum'
CUarray_format_enum__enumvalues = {
    1: 'CU_AD_FORMAT_UNSIGNED_INT8',
    2: 'CU_AD_FORMAT_UNSIGNED_INT16',
    3: 'CU_AD_FORMAT_UNSIGNED_INT32',
    8: 'CU_AD_FORMAT_SIGNED_INT8',
    9: 'CU_AD_FORMAT_SIGNED_INT16',
    10: 'CU_AD_FORMAT_SIGNED_INT32',
    16: 'CU_AD_FORMAT_HALF',
    32: 'CU_AD_FORMAT_FLOAT',
    176: 'CU_AD_FORMAT_NV12',
    192: 'CU_AD_FORMAT_UNORM_INT8X1',
    193: 'CU_AD_FORMAT_UNORM_INT8X2',
    194: 'CU_AD_FORMAT_UNORM_INT8X4',
    195: 'CU_AD_FORMAT_UNORM_INT16X1',
    196: 'CU_AD_FORMAT_UNORM_INT16X2',
    197: 'CU_AD_FORMAT_UNORM_INT16X4',
    198: 'CU_AD_FORMAT_SNORM_INT8X1',
    199: 'CU_AD_FORMAT_SNORM_INT8X2',
    200: 'CU_AD_FORMAT_SNORM_INT8X4',
    201: 'CU_AD_FORMAT_SNORM_INT16X1',
    202: 'CU_AD_FORMAT_SNORM_INT16X2',
    203: 'CU_AD_FORMAT_SNORM_INT16X4',
    145: 'CU_AD_FORMAT_BC1_UNORM',
    146: 'CU_AD_FORMAT_BC1_UNORM_SRGB',
    147: 'CU_AD_FORMAT_BC2_UNORM',
    148: 'CU_AD_FORMAT_BC2_UNORM_SRGB',
    149: 'CU_AD_FORMAT_BC3_UNORM',
    150: 'CU_AD_FORMAT_BC3_UNORM_SRGB',
    151: 'CU_AD_FORMAT_BC4_UNORM',
    152: 'CU_AD_FORMAT_BC4_SNORM',
    153: 'CU_AD_FORMAT_BC5_UNORM',
    154: 'CU_AD_FORMAT_BC5_SNORM',
    155: 'CU_AD_FORMAT_BC6H_UF16',
    156: 'CU_AD_FORMAT_BC6H_SF16',
    157: 'CU_AD_FORMAT_BC7_UNORM',
    158: 'CU_AD_FORMAT_BC7_UNORM_SRGB',
}
CU_AD_FORMAT_UNSIGNED_INT8 = 1
CU_AD_FORMAT_UNSIGNED_INT16 = 2
CU_AD_FORMAT_UNSIGNED_INT32 = 3
CU_AD_FORMAT_SIGNED_INT8 = 8
CU_AD_FORMAT_SIGNED_INT16 = 9
CU_AD_FORMAT_SIGNED_INT32 = 10
CU_AD_FORMAT_HALF = 16
CU_AD_FORMAT_FLOAT = 32
CU_AD_FORMAT_NV12 = 176
CU_AD_FORMAT_UNORM_INT8X1 = 192
CU_AD_FORMAT_UNORM_INT8X2 = 193
CU_AD_FORMAT_UNORM_INT8X4 = 194
CU_AD_FORMAT_UNORM_INT16X1 = 195
CU_AD_FORMAT_UNORM_INT16X2 = 196
CU_AD_FORMAT_UNORM_INT16X4 = 197
CU_AD_FORMAT_SNORM_INT8X1 = 198
CU_AD_FORMAT_SNORM_INT8X2 = 199
CU_AD_FORMAT_SNORM_INT8X4 = 200
CU_AD_FORMAT_SNORM_INT16X1 = 201
CU_AD_FORMAT_SNORM_INT16X2 = 202
CU_AD_FORMAT_SNORM_INT16X4 = 203
CU_AD_FORMAT_BC1_UNORM = 145
CU_AD_FORMAT_BC1_UNORM_SRGB = 146
CU_AD_FORMAT_BC2_UNORM = 147
CU_AD_FORMAT_BC2_UNORM_SRGB = 148
CU_AD_FORMAT_BC3_UNORM = 149
CU_AD_FORMAT_BC3_UNORM_SRGB = 150
CU_AD_FORMAT_BC4_UNORM = 151
CU_AD_FORMAT_BC4_SNORM = 152
CU_AD_FORMAT_BC5_UNORM = 153
CU_AD_FORMAT_BC5_SNORM = 154
CU_AD_FORMAT_BC6H_UF16 = 155
CU_AD_FORMAT_BC6H_SF16 = 156
CU_AD_FORMAT_BC7_UNORM = 157
CU_AD_FORMAT_BC7_UNORM_SRGB = 158
CUarray_format_enum = ctypes.c_uint32 # enum
CUarray_format = CUarray_format_enum
CUarray_format__enumvalues = CUarray_format_enum__enumvalues

# values for enumeration 'CUaddress_mode_enum'
CUaddress_mode_enum__enumvalues = {
    0: 'CU_TR_ADDRESS_MODE_WRAP',
    1: 'CU_TR_ADDRESS_MODE_CLAMP',
    2: 'CU_TR_ADDRESS_MODE_MIRROR',
    3: 'CU_TR_ADDRESS_MODE_BORDER',
}
CU_TR_ADDRESS_MODE_WRAP = 0
CU_TR_ADDRESS_MODE_CLAMP = 1
CU_TR_ADDRESS_MODE_MIRROR = 2
CU_TR_ADDRESS_MODE_BORDER = 3
CUaddress_mode_enum = ctypes.c_uint32 # enum
CUaddress_mode = CUaddress_mode_enum
CUaddress_mode__enumvalues = CUaddress_mode_enum__enumvalues

# values for enumeration 'CUfilter_mode_enum'
CUfilter_mode_enum__enumvalues = {
    0: 'CU_TR_FILTER_MODE_POINT',
    1: 'CU_TR_FILTER_MODE_LINEAR',
}
CU_TR_FILTER_MODE_POINT = 0
CU_TR_FILTER_MODE_LINEAR = 1
CUfilter_mode_enum = ctypes.c_uint32 # enum
CUfilter_mode = CUfilter_mode_enum
CUfilter_mode__enumvalues = CUfilter_mode_enum__enumvalues

# values for enumeration 'CUdevice_attribute_enum'
CUdevice_attribute_enum__enumvalues = {
    1: 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    2: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X',
    3: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y',
    4: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z',
    5: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X',
    6: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y',
    7: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z',
    8: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK',
    8: 'CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK',
    9: 'CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY',
    10: 'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    11: 'CU_DEVICE_ATTRIBUTE_MAX_PITCH',
    12: 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    12: 'CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK',
    13: 'CU_DEVICE_ATTRIBUTE_CLOCK_RATE',
    14: 'CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT',
    15: 'CU_DEVICE_ATTRIBUTE_GPU_OVERLAP',
    16: 'CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT',
    17: 'CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT',
    18: 'CU_DEVICE_ATTRIBUTE_INTEGRATED',
    19: 'CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY',
    20: 'CU_DEVICE_ATTRIBUTE_COMPUTE_MODE',
    21: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH',
    22: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH',
    23: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT',
    24: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH',
    25: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT',
    26: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH',
    27: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH',
    28: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT',
    29: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS',
    27: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH',
    28: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT',
    29: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES',
    30: 'CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT',
    31: 'CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS',
    32: 'CU_DEVICE_ATTRIBUTE_ECC_ENABLED',
    33: 'CU_DEVICE_ATTRIBUTE_PCI_BUS_ID',
    34: 'CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID',
    35: 'CU_DEVICE_ATTRIBUTE_TCC_DRIVER',
    36: 'CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE',
    37: 'CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH',
    38: 'CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE',
    39: 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    40: 'CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT',
    41: 'CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING',
    42: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH',
    43: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS',
    44: 'CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER',
    45: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH',
    46: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT',
    47: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE',
    48: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE',
    49: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE',
    50: 'CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID',
    51: 'CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT',
    52: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH',
    53: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH',
    54: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS',
    55: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH',
    56: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH',
    57: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT',
    58: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH',
    59: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT',
    60: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH',
    61: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH',
    62: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS',
    63: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH',
    64: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT',
    65: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS',
    66: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH',
    67: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH',
    68: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS',
    69: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH',
    70: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH',
    71: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT',
    72: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH',
    73: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH',
    74: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT',
    75: 'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR',
    76: 'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR',
    77: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH',
    78: 'CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED',
    79: 'CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED',
    80: 'CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED',
    81: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    82: 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    83: 'CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY',
    84: 'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD',
    85: 'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID',
    86: 'CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED',
    87: 'CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO',
    88: 'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS',
    89: 'CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS',
    90: 'CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED',
    91: 'CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM',
    92: 'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS',
    93: 'CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS',
    94: 'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR',
    95: 'CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH',
    96: 'CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH',
    97: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    98: 'CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES',
    99: 'CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED',
    100: 'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES',
    101: 'CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST',
    102: 'CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED',
    102: 'CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED',
    103: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED',
    104: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED',
    105: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED',
    106: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
    107: 'CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED',
    108: 'CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE',
    109: 'CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE',
    110: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED',
    111: 'CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK',
    112: 'CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED',
    113: 'CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED',
    114: 'CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED',
    115: 'CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED',
    116: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED',
    117: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS',
    118: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING',
    119: 'CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES',
    120: 'CU_DEVICE_ATTRIBUTE_MAX',
}
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105
CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106
CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107
CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108
CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110
CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111
CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113
CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114
CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
CU_DEVICE_ATTRIBUTE_MAX = 120
CUdevice_attribute_enum = ctypes.c_uint32 # enum
CUdevice_attribute = CUdevice_attribute_enum
CUdevice_attribute__enumvalues = CUdevice_attribute_enum__enumvalues
class struct_CUdevprop_st(Structure):
    pass

struct_CUdevprop_st._pack_ = 1 # source:False
struct_CUdevprop_st._fields_ = [
    ('maxThreadsPerBlock', ctypes.c_int32),
    ('maxThreadsDim', ctypes.c_int32 * 3),
    ('maxGridSize', ctypes.c_int32 * 3),
    ('sharedMemPerBlock', ctypes.c_int32),
    ('totalConstantMemory', ctypes.c_int32),
    ('SIMDWidth', ctypes.c_int32),
    ('memPitch', ctypes.c_int32),
    ('regsPerBlock', ctypes.c_int32),
    ('clockRate', ctypes.c_int32),
    ('textureAlign', ctypes.c_int32),
]

CUdevprop_v1 = struct_CUdevprop_st
CUdevprop = struct_CUdevprop_st

# values for enumeration 'CUpointer_attribute_enum'
CUpointer_attribute_enum__enumvalues = {
    1: 'CU_POINTER_ATTRIBUTE_CONTEXT',
    2: 'CU_POINTER_ATTRIBUTE_MEMORY_TYPE',
    3: 'CU_POINTER_ATTRIBUTE_DEVICE_POINTER',
    4: 'CU_POINTER_ATTRIBUTE_HOST_POINTER',
    5: 'CU_POINTER_ATTRIBUTE_P2P_TOKENS',
    6: 'CU_POINTER_ATTRIBUTE_SYNC_MEMOPS',
    7: 'CU_POINTER_ATTRIBUTE_BUFFER_ID',
    8: 'CU_POINTER_ATTRIBUTE_IS_MANAGED',
    9: 'CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    10: 'CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE',
    11: 'CU_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    12: 'CU_POINTER_ATTRIBUTE_RANGE_SIZE',
    13: 'CU_POINTER_ATTRIBUTE_MAPPED',
    14: 'CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    15: 'CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    16: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    17: 'CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
}
CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
CU_POINTER_ATTRIBUTE_BUFFER_ID = 7
CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
CU_POINTER_ATTRIBUTE_MAPPED = 13
CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
CUpointer_attribute_enum = ctypes.c_uint32 # enum
CUpointer_attribute = CUpointer_attribute_enum
CUpointer_attribute__enumvalues = CUpointer_attribute_enum__enumvalues

# values for enumeration 'CUfunction_attribute_enum'
CUfunction_attribute_enum__enumvalues = {
    0: 'CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    1: 'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    2: 'CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    3: 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
    4: 'CU_FUNC_ATTRIBUTE_NUM_REGS',
    5: 'CU_FUNC_ATTRIBUTE_PTX_VERSION',
    6: 'CU_FUNC_ATTRIBUTE_BINARY_VERSION',
    7: 'CU_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    8: 'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    9: 'CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    10: 'CU_FUNC_ATTRIBUTE_MAX',
}
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
CU_FUNC_ATTRIBUTE_MAX = 10
CUfunction_attribute_enum = ctypes.c_uint32 # enum
CUfunction_attribute = CUfunction_attribute_enum
CUfunction_attribute__enumvalues = CUfunction_attribute_enum__enumvalues

# values for enumeration 'CUfunc_cache_enum'
CUfunc_cache_enum__enumvalues = {
    0: 'CU_FUNC_CACHE_PREFER_NONE',
    1: 'CU_FUNC_CACHE_PREFER_SHARED',
    2: 'CU_FUNC_CACHE_PREFER_L1',
    3: 'CU_FUNC_CACHE_PREFER_EQUAL',
}
CU_FUNC_CACHE_PREFER_NONE = 0
CU_FUNC_CACHE_PREFER_SHARED = 1
CU_FUNC_CACHE_PREFER_L1 = 2
CU_FUNC_CACHE_PREFER_EQUAL = 3
CUfunc_cache_enum = ctypes.c_uint32 # enum
CUfunc_cache = CUfunc_cache_enum
CUfunc_cache__enumvalues = CUfunc_cache_enum__enumvalues

# values for enumeration 'CUsharedconfig_enum'
CUsharedconfig_enum__enumvalues = {
    0: 'CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE',
    1: 'CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE',
    2: 'CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE',
}
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
CUsharedconfig_enum = ctypes.c_uint32 # enum
CUsharedconfig = CUsharedconfig_enum
CUsharedconfig__enumvalues = CUsharedconfig_enum__enumvalues

# values for enumeration 'CUshared_carveout_enum'
CUshared_carveout_enum__enumvalues = {
    -1: 'CU_SHAREDMEM_CARVEOUT_DEFAULT',
    100: 'CU_SHAREDMEM_CARVEOUT_MAX_SHARED',
    0: 'CU_SHAREDMEM_CARVEOUT_MAX_L1',
}
CU_SHAREDMEM_CARVEOUT_DEFAULT = -1
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100
CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0
CUshared_carveout_enum = ctypes.c_int32 # enum
CUshared_carveout = CUshared_carveout_enum
CUshared_carveout__enumvalues = CUshared_carveout_enum__enumvalues

# values for enumeration 'CUmemorytype_enum'
CUmemorytype_enum__enumvalues = {
    1: 'CU_MEMORYTYPE_HOST',
    2: 'CU_MEMORYTYPE_DEVICE',
    3: 'CU_MEMORYTYPE_ARRAY',
    4: 'CU_MEMORYTYPE_UNIFIED',
}
CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3
CU_MEMORYTYPE_UNIFIED = 4
CUmemorytype_enum = ctypes.c_uint32 # enum
CUmemorytype = CUmemorytype_enum
CUmemorytype__enumvalues = CUmemorytype_enum__enumvalues

# values for enumeration 'CUcomputemode_enum'
CUcomputemode_enum__enumvalues = {
    0: 'CU_COMPUTEMODE_DEFAULT',
    2: 'CU_COMPUTEMODE_PROHIBITED',
    3: 'CU_COMPUTEMODE_EXCLUSIVE_PROCESS',
}
CU_COMPUTEMODE_DEFAULT = 0
CU_COMPUTEMODE_PROHIBITED = 2
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
CUcomputemode_enum = ctypes.c_uint32 # enum
CUcomputemode = CUcomputemode_enum
CUcomputemode__enumvalues = CUcomputemode_enum__enumvalues

# values for enumeration 'CUmem_advise_enum'
CUmem_advise_enum__enumvalues = {
    1: 'CU_MEM_ADVISE_SET_READ_MOSTLY',
    2: 'CU_MEM_ADVISE_UNSET_READ_MOSTLY',
    3: 'CU_MEM_ADVISE_SET_PREFERRED_LOCATION',
    4: 'CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION',
    5: 'CU_MEM_ADVISE_SET_ACCESSED_BY',
    6: 'CU_MEM_ADVISE_UNSET_ACCESSED_BY',
}
CU_MEM_ADVISE_SET_READ_MOSTLY = 1
CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4
CU_MEM_ADVISE_SET_ACCESSED_BY = 5
CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
CUmem_advise_enum = ctypes.c_uint32 # enum
CUmem_advise = CUmem_advise_enum
CUmem_advise__enumvalues = CUmem_advise_enum__enumvalues

# values for enumeration 'CUmem_range_attribute_enum'
CUmem_range_attribute_enum__enumvalues = {
    1: 'CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY',
    2: 'CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION',
    3: 'CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY',
    4: 'CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION',
}
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
CUmem_range_attribute_enum = ctypes.c_uint32 # enum
CUmem_range_attribute = CUmem_range_attribute_enum
CUmem_range_attribute__enumvalues = CUmem_range_attribute_enum__enumvalues

# values for enumeration 'CUjit_option_enum'
CUjit_option_enum__enumvalues = {
    0: 'CU_JIT_MAX_REGISTERS',
    1: 'CU_JIT_THREADS_PER_BLOCK',
    2: 'CU_JIT_WALL_TIME',
    3: 'CU_JIT_INFO_LOG_BUFFER',
    4: 'CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES',
    5: 'CU_JIT_ERROR_LOG_BUFFER',
    6: 'CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES',
    7: 'CU_JIT_OPTIMIZATION_LEVEL',
    8: 'CU_JIT_TARGET_FROM_CUCONTEXT',
    9: 'CU_JIT_TARGET',
    10: 'CU_JIT_FALLBACK_STRATEGY',
    11: 'CU_JIT_GENERATE_DEBUG_INFO',
    12: 'CU_JIT_LOG_VERBOSE',
    13: 'CU_JIT_GENERATE_LINE_INFO',
    14: 'CU_JIT_CACHE_MODE',
    15: 'CU_JIT_NEW_SM3X_OPT',
    16: 'CU_JIT_FAST_COMPILE',
    17: 'CU_JIT_GLOBAL_SYMBOL_NAMES',
    18: 'CU_JIT_GLOBAL_SYMBOL_ADDRESSES',
    19: 'CU_JIT_GLOBAL_SYMBOL_COUNT',
    20: 'CU_JIT_LTO',
    21: 'CU_JIT_FTZ',
    22: 'CU_JIT_PREC_DIV',
    23: 'CU_JIT_PREC_SQRT',
    24: 'CU_JIT_FMA',
    25: 'CU_JIT_NUM_OPTIONS',
}
CU_JIT_MAX_REGISTERS = 0
CU_JIT_THREADS_PER_BLOCK = 1
CU_JIT_WALL_TIME = 2
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_OPTIMIZATION_LEVEL = 7
CU_JIT_TARGET_FROM_CUCONTEXT = 8
CU_JIT_TARGET = 9
CU_JIT_FALLBACK_STRATEGY = 10
CU_JIT_GENERATE_DEBUG_INFO = 11
CU_JIT_LOG_VERBOSE = 12
CU_JIT_GENERATE_LINE_INFO = 13
CU_JIT_CACHE_MODE = 14
CU_JIT_NEW_SM3X_OPT = 15
CU_JIT_FAST_COMPILE = 16
CU_JIT_GLOBAL_SYMBOL_NAMES = 17
CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18
CU_JIT_GLOBAL_SYMBOL_COUNT = 19
CU_JIT_LTO = 20
CU_JIT_FTZ = 21
CU_JIT_PREC_DIV = 22
CU_JIT_PREC_SQRT = 23
CU_JIT_FMA = 24
CU_JIT_NUM_OPTIONS = 25
CUjit_option_enum = ctypes.c_uint32 # enum
CUjit_option = CUjit_option_enum
CUjit_option__enumvalues = CUjit_option_enum__enumvalues

# values for enumeration 'CUjit_target_enum'
CUjit_target_enum__enumvalues = {
    20: 'CU_TARGET_COMPUTE_20',
    21: 'CU_TARGET_COMPUTE_21',
    30: 'CU_TARGET_COMPUTE_30',
    32: 'CU_TARGET_COMPUTE_32',
    35: 'CU_TARGET_COMPUTE_35',
    37: 'CU_TARGET_COMPUTE_37',
    50: 'CU_TARGET_COMPUTE_50',
    52: 'CU_TARGET_COMPUTE_52',
    53: 'CU_TARGET_COMPUTE_53',
    60: 'CU_TARGET_COMPUTE_60',
    61: 'CU_TARGET_COMPUTE_61',
    62: 'CU_TARGET_COMPUTE_62',
    70: 'CU_TARGET_COMPUTE_70',
    72: 'CU_TARGET_COMPUTE_72',
    75: 'CU_TARGET_COMPUTE_75',
    80: 'CU_TARGET_COMPUTE_80',
    86: 'CU_TARGET_COMPUTE_86',
}
CU_TARGET_COMPUTE_20 = 20
CU_TARGET_COMPUTE_21 = 21
CU_TARGET_COMPUTE_30 = 30
CU_TARGET_COMPUTE_32 = 32
CU_TARGET_COMPUTE_35 = 35
CU_TARGET_COMPUTE_37 = 37
CU_TARGET_COMPUTE_50 = 50
CU_TARGET_COMPUTE_52 = 52
CU_TARGET_COMPUTE_53 = 53
CU_TARGET_COMPUTE_60 = 60
CU_TARGET_COMPUTE_61 = 61
CU_TARGET_COMPUTE_62 = 62
CU_TARGET_COMPUTE_70 = 70
CU_TARGET_COMPUTE_72 = 72
CU_TARGET_COMPUTE_75 = 75
CU_TARGET_COMPUTE_80 = 80
CU_TARGET_COMPUTE_86 = 86
CUjit_target_enum = ctypes.c_uint32 # enum
CUjit_target = CUjit_target_enum
CUjit_target__enumvalues = CUjit_target_enum__enumvalues

# values for enumeration 'CUjit_fallback_enum'
CUjit_fallback_enum__enumvalues = {
    0: 'CU_PREFER_PTX',
    1: 'CU_PREFER_BINARY',
}
CU_PREFER_PTX = 0
CU_PREFER_BINARY = 1
CUjit_fallback_enum = ctypes.c_uint32 # enum
CUjit_fallback = CUjit_fallback_enum
CUjit_fallback__enumvalues = CUjit_fallback_enum__enumvalues

# values for enumeration 'CUjit_cacheMode_enum'
CUjit_cacheMode_enum__enumvalues = {
    0: 'CU_JIT_CACHE_OPTION_NONE',
    1: 'CU_JIT_CACHE_OPTION_CG',
    2: 'CU_JIT_CACHE_OPTION_CA',
}
CU_JIT_CACHE_OPTION_NONE = 0
CU_JIT_CACHE_OPTION_CG = 1
CU_JIT_CACHE_OPTION_CA = 2
CUjit_cacheMode_enum = ctypes.c_uint32 # enum
CUjit_cacheMode = CUjit_cacheMode_enum
CUjit_cacheMode__enumvalues = CUjit_cacheMode_enum__enumvalues

# values for enumeration 'CUjitInputType_enum'
CUjitInputType_enum__enumvalues = {
    0: 'CU_JIT_INPUT_CUBIN',
    1: 'CU_JIT_INPUT_PTX',
    2: 'CU_JIT_INPUT_FATBINARY',
    3: 'CU_JIT_INPUT_OBJECT',
    4: 'CU_JIT_INPUT_LIBRARY',
    5: 'CU_JIT_INPUT_NVVM',
    6: 'CU_JIT_NUM_INPUT_TYPES',
}
CU_JIT_INPUT_CUBIN = 0
CU_JIT_INPUT_PTX = 1
CU_JIT_INPUT_FATBINARY = 2
CU_JIT_INPUT_OBJECT = 3
CU_JIT_INPUT_LIBRARY = 4
CU_JIT_INPUT_NVVM = 5
CU_JIT_NUM_INPUT_TYPES = 6
CUjitInputType_enum = ctypes.c_uint32 # enum
CUjitInputType = CUjitInputType_enum
CUjitInputType__enumvalues = CUjitInputType_enum__enumvalues
class struct_CUlinkState_st(Structure):
    pass

CUlinkState = ctypes.POINTER(struct_CUlinkState_st)

# values for enumeration 'CUgraphicsRegisterFlags_enum'
CUgraphicsRegisterFlags_enum__enumvalues = {
    0: 'CU_GRAPHICS_REGISTER_FLAGS_NONE',
    1: 'CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY',
    2: 'CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD',
    4: 'CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST',
    8: 'CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER',
}
CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8
CUgraphicsRegisterFlags_enum = ctypes.c_uint32 # enum
CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum
CUgraphicsRegisterFlags__enumvalues = CUgraphicsRegisterFlags_enum__enumvalues

# values for enumeration 'CUgraphicsMapResourceFlags_enum'
CUgraphicsMapResourceFlags_enum__enumvalues = {
    0: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE',
    1: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY',
    2: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD',
}
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2
CUgraphicsMapResourceFlags_enum = ctypes.c_uint32 # enum
CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum
CUgraphicsMapResourceFlags__enumvalues = CUgraphicsMapResourceFlags_enum__enumvalues

# values for enumeration 'CUarray_cubemap_face_enum'
CUarray_cubemap_face_enum__enumvalues = {
    0: 'CU_CUBEMAP_FACE_POSITIVE_X',
    1: 'CU_CUBEMAP_FACE_NEGATIVE_X',
    2: 'CU_CUBEMAP_FACE_POSITIVE_Y',
    3: 'CU_CUBEMAP_FACE_NEGATIVE_Y',
    4: 'CU_CUBEMAP_FACE_POSITIVE_Z',
    5: 'CU_CUBEMAP_FACE_NEGATIVE_Z',
}
CU_CUBEMAP_FACE_POSITIVE_X = 0
CU_CUBEMAP_FACE_NEGATIVE_X = 1
CU_CUBEMAP_FACE_POSITIVE_Y = 2
CU_CUBEMAP_FACE_NEGATIVE_Y = 3
CU_CUBEMAP_FACE_POSITIVE_Z = 4
CU_CUBEMAP_FACE_NEGATIVE_Z = 5
CUarray_cubemap_face_enum = ctypes.c_uint32 # enum
CUarray_cubemap_face = CUarray_cubemap_face_enum
CUarray_cubemap_face__enumvalues = CUarray_cubemap_face_enum__enumvalues

# values for enumeration 'CUlimit_enum'
CUlimit_enum__enumvalues = {
    0: 'CU_LIMIT_STACK_SIZE',
    1: 'CU_LIMIT_PRINTF_FIFO_SIZE',
    2: 'CU_LIMIT_MALLOC_HEAP_SIZE',
    3: 'CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH',
    4: 'CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT',
    5: 'CU_LIMIT_MAX_L2_FETCH_GRANULARITY',
    6: 'CU_LIMIT_PERSISTING_L2_CACHE_SIZE',
    7: 'CU_LIMIT_MAX',
}
CU_LIMIT_STACK_SIZE = 0
CU_LIMIT_PRINTF_FIFO_SIZE = 1
CU_LIMIT_MALLOC_HEAP_SIZE = 2
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4
CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5
CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6
CU_LIMIT_MAX = 7
CUlimit_enum = ctypes.c_uint32 # enum
CUlimit = CUlimit_enum
CUlimit__enumvalues = CUlimit_enum__enumvalues

# values for enumeration 'CUresourcetype_enum'
CUresourcetype_enum__enumvalues = {
    0: 'CU_RESOURCE_TYPE_ARRAY',
    1: 'CU_RESOURCE_TYPE_MIPMAPPED_ARRAY',
    2: 'CU_RESOURCE_TYPE_LINEAR',
    3: 'CU_RESOURCE_TYPE_PITCH2D',
}
CU_RESOURCE_TYPE_ARRAY = 0
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
CU_RESOURCE_TYPE_LINEAR = 2
CU_RESOURCE_TYPE_PITCH2D = 3
CUresourcetype_enum = ctypes.c_uint32 # enum
CUresourcetype = CUresourcetype_enum
CUresourcetype__enumvalues = CUresourcetype_enum__enumvalues
CUhostFn = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))

# values for enumeration 'CUaccessProperty_enum'
CUaccessProperty_enum__enumvalues = {
    0: 'CU_ACCESS_PROPERTY_NORMAL',
    1: 'CU_ACCESS_PROPERTY_STREAMING',
    2: 'CU_ACCESS_PROPERTY_PERSISTING',
}
CU_ACCESS_PROPERTY_NORMAL = 0
CU_ACCESS_PROPERTY_STREAMING = 1
CU_ACCESS_PROPERTY_PERSISTING = 2
CUaccessProperty_enum = ctypes.c_uint32 # enum
CUaccessProperty = CUaccessProperty_enum
CUaccessProperty__enumvalues = CUaccessProperty_enum__enumvalues
class struct_CUaccessPolicyWindow_st(Structure):
    pass

struct_CUaccessPolicyWindow_st._pack_ = 1 # source:False
struct_CUaccessPolicyWindow_st._fields_ = [
    ('base_ptr', ctypes.POINTER(None)),
    ('num_bytes', ctypes.c_uint64),
    ('hitRatio', ctypes.c_float),
    ('hitProp', CUaccessProperty),
    ('missProp', CUaccessProperty),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUaccessPolicyWindow_v1 = struct_CUaccessPolicyWindow_st
CUaccessPolicyWindow = struct_CUaccessPolicyWindow_st
class struct_CUDA_KERNEL_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_KERNEL_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_KERNEL_NODE_PARAMS_st._fields_ = [
    ('func', ctypes.POINTER(struct_CUfunc_st)),
    ('gridDimX', ctypes.c_uint32),
    ('gridDimY', ctypes.c_uint32),
    ('gridDimZ', ctypes.c_uint32),
    ('blockDimX', ctypes.c_uint32),
    ('blockDimY', ctypes.c_uint32),
    ('blockDimZ', ctypes.c_uint32),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
    ('extra', ctypes.POINTER(ctypes.POINTER(None))),
]

CUDA_KERNEL_NODE_PARAMS_v1 = struct_CUDA_KERNEL_NODE_PARAMS_st
CUDA_KERNEL_NODE_PARAMS = struct_CUDA_KERNEL_NODE_PARAMS_st
class struct_CUDA_MEMSET_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_MEMSET_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_MEMSET_NODE_PARAMS_st._fields_ = [
    ('dst', ctypes.c_uint64),
    ('pitch', ctypes.c_uint64),
    ('value', ctypes.c_uint32),
    ('elementSize', ctypes.c_uint32),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
]

CUDA_MEMSET_NODE_PARAMS_v1 = struct_CUDA_MEMSET_NODE_PARAMS_st
CUDA_MEMSET_NODE_PARAMS = struct_CUDA_MEMSET_NODE_PARAMS_st
class struct_CUDA_HOST_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_HOST_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_HOST_NODE_PARAMS_st._fields_ = [
    ('fn', ctypes.CFUNCTYPE(None, ctypes.POINTER(None))),
    ('userData', ctypes.POINTER(None)),
]

CUDA_HOST_NODE_PARAMS_v1 = struct_CUDA_HOST_NODE_PARAMS_st
CUDA_HOST_NODE_PARAMS = struct_CUDA_HOST_NODE_PARAMS_st

# values for enumeration 'CUgraphNodeType_enum'
CUgraphNodeType_enum__enumvalues = {
    0: 'CU_GRAPH_NODE_TYPE_KERNEL',
    1: 'CU_GRAPH_NODE_TYPE_MEMCPY',
    2: 'CU_GRAPH_NODE_TYPE_MEMSET',
    3: 'CU_GRAPH_NODE_TYPE_HOST',
    4: 'CU_GRAPH_NODE_TYPE_GRAPH',
    5: 'CU_GRAPH_NODE_TYPE_EMPTY',
    6: 'CU_GRAPH_NODE_TYPE_WAIT_EVENT',
    7: 'CU_GRAPH_NODE_TYPE_EVENT_RECORD',
    8: 'CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL',
    9: 'CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT',
    10: 'CU_GRAPH_NODE_TYPE_MEM_ALLOC',
    11: 'CU_GRAPH_NODE_TYPE_MEM_FREE',
}
CU_GRAPH_NODE_TYPE_KERNEL = 0
CU_GRAPH_NODE_TYPE_MEMCPY = 1
CU_GRAPH_NODE_TYPE_MEMSET = 2
CU_GRAPH_NODE_TYPE_HOST = 3
CU_GRAPH_NODE_TYPE_GRAPH = 4
CU_GRAPH_NODE_TYPE_EMPTY = 5
CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6
CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7
CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9
CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10
CU_GRAPH_NODE_TYPE_MEM_FREE = 11
CUgraphNodeType_enum = ctypes.c_uint32 # enum
CUgraphNodeType = CUgraphNodeType_enum
CUgraphNodeType__enumvalues = CUgraphNodeType_enum__enumvalues

# values for enumeration 'CUsynchronizationPolicy_enum'
CUsynchronizationPolicy_enum__enumvalues = {
    1: 'CU_SYNC_POLICY_AUTO',
    2: 'CU_SYNC_POLICY_SPIN',
    3: 'CU_SYNC_POLICY_YIELD',
    4: 'CU_SYNC_POLICY_BLOCKING_SYNC',
}
CU_SYNC_POLICY_AUTO = 1
CU_SYNC_POLICY_SPIN = 2
CU_SYNC_POLICY_YIELD = 3
CU_SYNC_POLICY_BLOCKING_SYNC = 4
CUsynchronizationPolicy_enum = ctypes.c_uint32 # enum
CUsynchronizationPolicy = CUsynchronizationPolicy_enum
CUsynchronizationPolicy__enumvalues = CUsynchronizationPolicy_enum__enumvalues

# values for enumeration 'CUkernelNodeAttrID_enum'
CUkernelNodeAttrID_enum__enumvalues = {
    1: 'CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    2: 'CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE',
}
CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2
CUkernelNodeAttrID_enum = ctypes.c_uint32 # enum
CUkernelNodeAttrID = CUkernelNodeAttrID_enum
CUkernelNodeAttrID__enumvalues = CUkernelNodeAttrID_enum__enumvalues
class union_CUkernelNodeAttrValue_union(Union):
    pass

union_CUkernelNodeAttrValue_union._pack_ = 1 # source:False
union_CUkernelNodeAttrValue_union._fields_ = [
    ('accessPolicyWindow', CUaccessPolicyWindow),
    ('cooperative', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

CUkernelNodeAttrValue_v1 = union_CUkernelNodeAttrValue_union
CUkernelNodeAttrValue = union_CUkernelNodeAttrValue_union

# values for enumeration 'CUstreamCaptureStatus_enum'
CUstreamCaptureStatus_enum__enumvalues = {
    0: 'CU_STREAM_CAPTURE_STATUS_NONE',
    1: 'CU_STREAM_CAPTURE_STATUS_ACTIVE',
    2: 'CU_STREAM_CAPTURE_STATUS_INVALIDATED',
}
CU_STREAM_CAPTURE_STATUS_NONE = 0
CU_STREAM_CAPTURE_STATUS_ACTIVE = 1
CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
CUstreamCaptureStatus_enum = ctypes.c_uint32 # enum
CUstreamCaptureStatus = CUstreamCaptureStatus_enum
CUstreamCaptureStatus__enumvalues = CUstreamCaptureStatus_enum__enumvalues

# values for enumeration 'CUstreamCaptureMode_enum'
CUstreamCaptureMode_enum__enumvalues = {
    0: 'CU_STREAM_CAPTURE_MODE_GLOBAL',
    1: 'CU_STREAM_CAPTURE_MODE_THREAD_LOCAL',
    2: 'CU_STREAM_CAPTURE_MODE_RELAXED',
}
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
CU_STREAM_CAPTURE_MODE_RELAXED = 2
CUstreamCaptureMode_enum = ctypes.c_uint32 # enum
CUstreamCaptureMode = CUstreamCaptureMode_enum
CUstreamCaptureMode__enumvalues = CUstreamCaptureMode_enum__enumvalues

# values for enumeration 'CUstreamAttrID_enum'
CUstreamAttrID_enum__enumvalues = {
    1: 'CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    3: 'CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY',
}
CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
CUstreamAttrID_enum = ctypes.c_uint32 # enum
CUstreamAttrID = CUstreamAttrID_enum
CUstreamAttrID__enumvalues = CUstreamAttrID_enum__enumvalues
class union_CUstreamAttrValue_union(Union):
    pass

union_CUstreamAttrValue_union._pack_ = 1 # source:False
union_CUstreamAttrValue_union._fields_ = [
    ('accessPolicyWindow', CUaccessPolicyWindow),
    ('syncPolicy', CUsynchronizationPolicy),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

CUstreamAttrValue_v1 = union_CUstreamAttrValue_union
CUstreamAttrValue = union_CUstreamAttrValue_union

# values for enumeration 'CUdriverProcAddress_flags_enum'
CUdriverProcAddress_flags_enum__enumvalues = {
    0: 'CU_GET_PROC_ADDRESS_DEFAULT',
    1: 'CU_GET_PROC_ADDRESS_LEGACY_STREAM',
    2: 'CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM',
}
CU_GET_PROC_ADDRESS_DEFAULT = 0
CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1
CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2
CUdriverProcAddress_flags_enum = ctypes.c_uint32 # enum
CUdriverProcAddress_flags = CUdriverProcAddress_flags_enum
CUdriverProcAddress_flags__enumvalues = CUdriverProcAddress_flags_enum__enumvalues

# values for enumeration 'CUexecAffinityType_enum'
CUexecAffinityType_enum__enumvalues = {
    0: 'CU_EXEC_AFFINITY_TYPE_SM_COUNT',
    1: 'CU_EXEC_AFFINITY_TYPE_MAX',
}
CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0
CU_EXEC_AFFINITY_TYPE_MAX = 1
CUexecAffinityType_enum = ctypes.c_uint32 # enum
CUexecAffinityType = CUexecAffinityType_enum
CUexecAffinityType__enumvalues = CUexecAffinityType_enum__enumvalues
class struct_CUexecAffinitySmCount_st(Structure):
    pass

struct_CUexecAffinitySmCount_st._pack_ = 1 # source:False
struct_CUexecAffinitySmCount_st._fields_ = [
    ('val', ctypes.c_uint32),
]

CUexecAffinitySmCount_v1 = struct_CUexecAffinitySmCount_st
CUexecAffinitySmCount = struct_CUexecAffinitySmCount_st
class struct_CUexecAffinityParam_st(Structure):
    pass

class union_CUexecAffinityParam_st_param(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('smCount', CUexecAffinitySmCount),
     ]

struct_CUexecAffinityParam_st._pack_ = 1 # source:False
struct_CUexecAffinityParam_st._fields_ = [
    ('type', CUexecAffinityType),
    ('param', union_CUexecAffinityParam_st_param),
]

CUexecAffinityParam_v1 = struct_CUexecAffinityParam_st
CUexecAffinityParam = struct_CUexecAffinityParam_st

# values for enumeration 'cudaError_enum'
cudaError_enum__enumvalues = {
    0: 'CUDA_SUCCESS',
    1: 'CUDA_ERROR_INVALID_VALUE',
    2: 'CUDA_ERROR_OUT_OF_MEMORY',
    3: 'CUDA_ERROR_NOT_INITIALIZED',
    4: 'CUDA_ERROR_DEINITIALIZED',
    5: 'CUDA_ERROR_PROFILER_DISABLED',
    6: 'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    7: 'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    8: 'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    34: 'CUDA_ERROR_STUB_LIBRARY',
    100: 'CUDA_ERROR_NO_DEVICE',
    101: 'CUDA_ERROR_INVALID_DEVICE',
    102: 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    200: 'CUDA_ERROR_INVALID_IMAGE',
    201: 'CUDA_ERROR_INVALID_CONTEXT',
    202: 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    205: 'CUDA_ERROR_MAP_FAILED',
    206: 'CUDA_ERROR_UNMAP_FAILED',
    207: 'CUDA_ERROR_ARRAY_IS_MAPPED',
    208: 'CUDA_ERROR_ALREADY_MAPPED',
    209: 'CUDA_ERROR_NO_BINARY_FOR_GPU',
    210: 'CUDA_ERROR_ALREADY_ACQUIRED',
    211: 'CUDA_ERROR_NOT_MAPPED',
    212: 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    213: 'CUDA_ERROR_NOT_MAPPED_AS_POINTER',
    214: 'CUDA_ERROR_ECC_UNCORRECTABLE',
    215: 'CUDA_ERROR_UNSUPPORTED_LIMIT',
    216: 'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    217: 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    218: 'CUDA_ERROR_INVALID_PTX',
    219: 'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    220: 'CUDA_ERROR_NVLINK_UNCORRECTABLE',
    221: 'CUDA_ERROR_JIT_COMPILER_NOT_FOUND',
    222: 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
    223: 'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    224: 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    300: 'CUDA_ERROR_INVALID_SOURCE',
    301: 'CUDA_ERROR_FILE_NOT_FOUND',
    302: 'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    303: 'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    304: 'CUDA_ERROR_OPERATING_SYSTEM',
    400: 'CUDA_ERROR_INVALID_HANDLE',
    401: 'CUDA_ERROR_ILLEGAL_STATE',
    500: 'CUDA_ERROR_NOT_FOUND',
    600: 'CUDA_ERROR_NOT_READY',
    700: 'CUDA_ERROR_ILLEGAL_ADDRESS',
    701: 'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES',
    702: 'CUDA_ERROR_LAUNCH_TIMEOUT',
    703: 'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    704: 'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    705: 'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    708: 'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    709: 'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    710: 'CUDA_ERROR_ASSERT',
    711: 'CUDA_ERROR_TOO_MANY_PEERS',
    712: 'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    713: 'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    714: 'CUDA_ERROR_HARDWARE_STACK_ERROR',
    715: 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    716: 'CUDA_ERROR_MISALIGNED_ADDRESS',
    717: 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    718: 'CUDA_ERROR_INVALID_PC',
    719: 'CUDA_ERROR_LAUNCH_FAILED',
    720: 'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    800: 'CUDA_ERROR_NOT_PERMITTED',
    801: 'CUDA_ERROR_NOT_SUPPORTED',
    802: 'CUDA_ERROR_SYSTEM_NOT_READY',
    803: 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    804: 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    805: 'CUDA_ERROR_MPS_CONNECTION_FAILED',
    806: 'CUDA_ERROR_MPS_RPC_FAILURE',
    807: 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    808: 'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    809: 'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    900: 'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    901: 'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    902: 'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    903: 'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    904: 'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    905: 'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    906: 'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    907: 'CUDA_ERROR_CAPTURED_EVENT',
    908: 'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    909: 'CUDA_ERROR_TIMEOUT',
    910: 'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    911: 'CUDA_ERROR_EXTERNAL_DEVICE',
    999: 'CUDA_ERROR_UNKNOWN',
}
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_PROFILER_DISABLED = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
CUDA_ERROR_STUB_LIBRARY = 34
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_DEVICE_NOT_LICENSED = 102
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED = 205
CUDA_ERROR_UNMAP_FAILED = 206
CUDA_ERROR_ARRAY_IS_MAPPED = 207
CUDA_ERROR_ALREADY_MAPPED = 208
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_ALREADY_ACQUIRED = 210
CUDA_ERROR_NOT_MAPPED = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
CUDA_ERROR_ECC_UNCORRECTABLE = 214
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_FILE_NOT_FOUND = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
CUDA_ERROR_OPERATING_SYSTEM = 304
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_ILLEGAL_STATE = 401
CUDA_ERROR_NOT_FOUND = 500
CUDA_ERROR_NOT_READY = 600
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_TOO_MANY_PEERS = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
CUDA_ERROR_INVALID_PC = 718
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
CUDA_ERROR_NOT_PERMITTED = 800
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_SYSTEM_NOT_READY = 802
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
CUDA_ERROR_MPS_CONNECTION_FAILED = 805
CUDA_ERROR_MPS_RPC_FAILURE = 806
CUDA_ERROR_MPS_SERVER_NOT_READY = 807
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
CUDA_ERROR_CAPTURED_EVENT = 907
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
CUDA_ERROR_TIMEOUT = 909
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
CUDA_ERROR_EXTERNAL_DEVICE = 911
CUDA_ERROR_UNKNOWN = 999
cudaError_enum = ctypes.c_uint32 # enum
CUresult = cudaError_enum
CUresult__enumvalues = cudaError_enum__enumvalues

# values for enumeration 'CUdevice_P2PAttribute_enum'
CUdevice_P2PAttribute_enum__enumvalues = {
    1: 'CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK',
    2: 'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED',
    3: 'CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED',
    4: 'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED',
    4: 'CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED',
}
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
CUdevice_P2PAttribute_enum = ctypes.c_uint32 # enum
CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum
CUdevice_P2PAttribute__enumvalues = CUdevice_P2PAttribute_enum__enumvalues
CUstreamCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_CUstream_st), cudaError_enum, ctypes.POINTER(None))
CUoccupancyB2DSize = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_int32)
class struct_CUDA_MEMCPY2D_st(Structure):
    pass

struct_CUDA_MEMCPY2D_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY2D_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('srcPitch', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('dstPitch', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
]

CUDA_MEMCPY2D_v2 = struct_CUDA_MEMCPY2D_st
CUDA_MEMCPY2D = struct_CUDA_MEMCPY2D_st
class struct_CUDA_MEMCPY3D_st(Structure):
    pass

struct_CUDA_MEMCPY3D_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY3D_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcZ', ctypes.c_uint64),
    ('srcLOD', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('reserved0', ctypes.POINTER(None)),
    ('srcPitch', ctypes.c_uint64),
    ('srcHeight', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstZ', ctypes.c_uint64),
    ('dstLOD', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('reserved1', ctypes.POINTER(None)),
    ('dstPitch', ctypes.c_uint64),
    ('dstHeight', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
]

CUDA_MEMCPY3D_v2 = struct_CUDA_MEMCPY3D_st
CUDA_MEMCPY3D = struct_CUDA_MEMCPY3D_st
class struct_CUDA_MEMCPY3D_PEER_st(Structure):
    pass

struct_CUDA_MEMCPY3D_PEER_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY3D_PEER_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcZ', ctypes.c_uint64),
    ('srcLOD', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('srcContext', ctypes.POINTER(struct_CUctx_st)),
    ('srcPitch', ctypes.c_uint64),
    ('srcHeight', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstZ', ctypes.c_uint64),
    ('dstLOD', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('dstContext', ctypes.POINTER(struct_CUctx_st)),
    ('dstPitch', ctypes.c_uint64),
    ('dstHeight', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
]

CUDA_MEMCPY3D_PEER_v1 = struct_CUDA_MEMCPY3D_PEER_st
CUDA_MEMCPY3D_PEER = struct_CUDA_MEMCPY3D_PEER_st
class struct_CUDA_ARRAY_DESCRIPTOR_st(Structure):
    pass

struct_CUDA_ARRAY_DESCRIPTOR_st._pack_ = 1 # source:False
struct_CUDA_ARRAY_DESCRIPTOR_st._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Format', CUarray_format),
    ('NumChannels', ctypes.c_uint32),
]

CUDA_ARRAY_DESCRIPTOR_v2 = struct_CUDA_ARRAY_DESCRIPTOR_st
CUDA_ARRAY_DESCRIPTOR = struct_CUDA_ARRAY_DESCRIPTOR_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_st(Structure):
    pass

struct_CUDA_ARRAY3D_DESCRIPTOR_st._pack_ = 1 # source:False
struct_CUDA_ARRAY3D_DESCRIPTOR_st._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
    ('Format', CUarray_format),
    ('NumChannels', ctypes.c_uint32),
    ('Flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_ARRAY3D_DESCRIPTOR_v2 = struct_CUDA_ARRAY3D_DESCRIPTOR_st
CUDA_ARRAY3D_DESCRIPTOR = struct_CUDA_ARRAY3D_DESCRIPTOR_st
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st(Structure):
    pass

class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent(Structure):
    pass

struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent._pack_ = 1 # source:False
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
]

struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._pack_ = 1 # source:False
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._fields_ = [
    ('tileExtent', struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent),
    ('miptailFirstLevel', ctypes.c_uint32),
    ('miptailSize', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 4),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_ARRAY_SPARSE_PROPERTIES_v1 = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
CUDA_ARRAY_SPARSE_PROPERTIES = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
class struct_CUDA_RESOURCE_DESC_st(Structure):
    pass

class union_CUDA_RESOURCE_DESC_st_res(Union):
    pass

class struct_CUDA_RESOURCE_DESC_st_0_array(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_array._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_array._fields_ = [
    ('hArray', ctypes.POINTER(struct_CUarray_st)),
]

class struct_CUDA_RESOURCE_DESC_st_0_mipmap(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_mipmap._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_mipmap._fields_ = [
    ('hMipmappedArray', ctypes.POINTER(struct_CUmipmappedArray_st)),
]

class struct_CUDA_RESOURCE_DESC_st_0_linear(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_linear._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_linear._fields_ = [
    ('devPtr', ctypes.c_uint64),
    ('format', CUarray_format),
    ('numChannels', ctypes.c_uint32),
    ('sizeInBytes', ctypes.c_uint64),
]

class struct_CUDA_RESOURCE_DESC_st_0_pitch2D(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_pitch2D._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_pitch2D._fields_ = [
    ('devPtr', ctypes.c_uint64),
    ('format', CUarray_format),
    ('numChannels', ctypes.c_uint32),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('pitchInBytes', ctypes.c_uint64),
]

class struct_CUDA_RESOURCE_DESC_st_0_reserved(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_reserved._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_reserved._fields_ = [
    ('reserved', ctypes.c_int32 * 32),
]

union_CUDA_RESOURCE_DESC_st_res._pack_ = 1 # source:False
union_CUDA_RESOURCE_DESC_st_res._fields_ = [
    ('array', struct_CUDA_RESOURCE_DESC_st_0_array),
    ('mipmap', struct_CUDA_RESOURCE_DESC_st_0_mipmap),
    ('linear', struct_CUDA_RESOURCE_DESC_st_0_linear),
    ('pitch2D', struct_CUDA_RESOURCE_DESC_st_0_pitch2D),
    ('reserved', struct_CUDA_RESOURCE_DESC_st_0_reserved),
]

struct_CUDA_RESOURCE_DESC_st._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st._fields_ = [
    ('resType', CUresourcetype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('res', union_CUDA_RESOURCE_DESC_st_res),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_RESOURCE_DESC_v1 = struct_CUDA_RESOURCE_DESC_st
CUDA_RESOURCE_DESC = struct_CUDA_RESOURCE_DESC_st
class struct_CUDA_TEXTURE_DESC_st(Structure):
    pass

struct_CUDA_TEXTURE_DESC_st._pack_ = 1 # source:False
struct_CUDA_TEXTURE_DESC_st._fields_ = [
    ('addressMode', CUaddress_mode_enum * 3),
    ('filterMode', CUfilter_mode),
    ('flags', ctypes.c_uint32),
    ('maxAnisotropy', ctypes.c_uint32),
    ('mipmapFilterMode', CUfilter_mode),
    ('mipmapLevelBias', ctypes.c_float),
    ('minMipmapLevelClamp', ctypes.c_float),
    ('maxMipmapLevelClamp', ctypes.c_float),
    ('borderColor', ctypes.c_float * 4),
    ('reserved', ctypes.c_int32 * 12),
]

CUDA_TEXTURE_DESC_v1 = struct_CUDA_TEXTURE_DESC_st
CUDA_TEXTURE_DESC = struct_CUDA_TEXTURE_DESC_st

# values for enumeration 'CUresourceViewFormat_enum'
CUresourceViewFormat_enum__enumvalues = {
    0: 'CU_RES_VIEW_FORMAT_NONE',
    1: 'CU_RES_VIEW_FORMAT_UINT_1X8',
    2: 'CU_RES_VIEW_FORMAT_UINT_2X8',
    3: 'CU_RES_VIEW_FORMAT_UINT_4X8',
    4: 'CU_RES_VIEW_FORMAT_SINT_1X8',
    5: 'CU_RES_VIEW_FORMAT_SINT_2X8',
    6: 'CU_RES_VIEW_FORMAT_SINT_4X8',
    7: 'CU_RES_VIEW_FORMAT_UINT_1X16',
    8: 'CU_RES_VIEW_FORMAT_UINT_2X16',
    9: 'CU_RES_VIEW_FORMAT_UINT_4X16',
    10: 'CU_RES_VIEW_FORMAT_SINT_1X16',
    11: 'CU_RES_VIEW_FORMAT_SINT_2X16',
    12: 'CU_RES_VIEW_FORMAT_SINT_4X16',
    13: 'CU_RES_VIEW_FORMAT_UINT_1X32',
    14: 'CU_RES_VIEW_FORMAT_UINT_2X32',
    15: 'CU_RES_VIEW_FORMAT_UINT_4X32',
    16: 'CU_RES_VIEW_FORMAT_SINT_1X32',
    17: 'CU_RES_VIEW_FORMAT_SINT_2X32',
    18: 'CU_RES_VIEW_FORMAT_SINT_4X32',
    19: 'CU_RES_VIEW_FORMAT_FLOAT_1X16',
    20: 'CU_RES_VIEW_FORMAT_FLOAT_2X16',
    21: 'CU_RES_VIEW_FORMAT_FLOAT_4X16',
    22: 'CU_RES_VIEW_FORMAT_FLOAT_1X32',
    23: 'CU_RES_VIEW_FORMAT_FLOAT_2X32',
    24: 'CU_RES_VIEW_FORMAT_FLOAT_4X32',
    25: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC1',
    26: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC2',
    27: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC3',
    28: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC4',
    29: 'CU_RES_VIEW_FORMAT_SIGNED_BC4',
    30: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC5',
    31: 'CU_RES_VIEW_FORMAT_SIGNED_BC5',
    32: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    33: 'CU_RES_VIEW_FORMAT_SIGNED_BC6H',
    34: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC7',
}
CU_RES_VIEW_FORMAT_NONE = 0
CU_RES_VIEW_FORMAT_UINT_1X8 = 1
CU_RES_VIEW_FORMAT_UINT_2X8 = 2
CU_RES_VIEW_FORMAT_UINT_4X8 = 3
CU_RES_VIEW_FORMAT_SINT_1X8 = 4
CU_RES_VIEW_FORMAT_SINT_2X8 = 5
CU_RES_VIEW_FORMAT_SINT_4X8 = 6
CU_RES_VIEW_FORMAT_UINT_1X16 = 7
CU_RES_VIEW_FORMAT_UINT_2X16 = 8
CU_RES_VIEW_FORMAT_UINT_4X16 = 9
CU_RES_VIEW_FORMAT_SINT_1X16 = 10
CU_RES_VIEW_FORMAT_SINT_2X16 = 11
CU_RES_VIEW_FORMAT_SINT_4X16 = 12
CU_RES_VIEW_FORMAT_UINT_1X32 = 13
CU_RES_VIEW_FORMAT_UINT_2X32 = 14
CU_RES_VIEW_FORMAT_UINT_4X32 = 15
CU_RES_VIEW_FORMAT_SINT_1X32 = 16
CU_RES_VIEW_FORMAT_SINT_2X32 = 17
CU_RES_VIEW_FORMAT_SINT_4X32 = 18
CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19
CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20
CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21
CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22
CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23
CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24
CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29
CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33
CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
CUresourceViewFormat_enum = ctypes.c_uint32 # enum
CUresourceViewFormat = CUresourceViewFormat_enum
CUresourceViewFormat__enumvalues = CUresourceViewFormat_enum__enumvalues
class struct_CUDA_RESOURCE_VIEW_DESC_st(Structure):
    pass

struct_CUDA_RESOURCE_VIEW_DESC_st._pack_ = 1 # source:False
struct_CUDA_RESOURCE_VIEW_DESC_st._fields_ = [
    ('format', CUresourceViewFormat),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('firstMipmapLevel', ctypes.c_uint32),
    ('lastMipmapLevel', ctypes.c_uint32),
    ('firstLayer', ctypes.c_uint32),
    ('lastLayer', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
]

CUDA_RESOURCE_VIEW_DESC_v1 = struct_CUDA_RESOURCE_VIEW_DESC_st
CUDA_RESOURCE_VIEW_DESC = struct_CUDA_RESOURCE_VIEW_DESC_st
class struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st(Structure):
    pass

struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._pack_ = 1 # source:False
struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._fields_ = [
    ('p2pToken', ctypes.c_uint64),
    ('vaSpaceToken', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st

# values for enumeration 'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum'
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum__enumvalues = {
    0: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE',
    1: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ',
    3: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE',
}
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = ctypes.c_uint32 # enum
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS__enumvalues = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum__enumvalues
class struct_CUDA_LAUNCH_PARAMS_st(Structure):
    pass

struct_CUDA_LAUNCH_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_LAUNCH_PARAMS_st._fields_ = [
    ('function', ctypes.POINTER(struct_CUfunc_st)),
    ('gridDimX', ctypes.c_uint32),
    ('gridDimY', ctypes.c_uint32),
    ('gridDimZ', ctypes.c_uint32),
    ('blockDimX', ctypes.c_uint32),
    ('blockDimY', ctypes.c_uint32),
    ('blockDimZ', ctypes.c_uint32),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('hStream', ctypes.POINTER(struct_CUstream_st)),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
]

CUDA_LAUNCH_PARAMS_v1 = struct_CUDA_LAUNCH_PARAMS_st
CUDA_LAUNCH_PARAMS = struct_CUDA_LAUNCH_PARAMS_st

# values for enumeration 'CUexternalMemoryHandleType_enum'
CUexternalMemoryHandleType_enum__enumvalues = {
    1: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD',
    2: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32',
    3: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    4: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP',
    5: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE',
    6: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE',
    7: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT',
    8: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF',
}
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
CUexternalMemoryHandleType_enum = ctypes.c_uint32 # enum
CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum
CUexternalMemoryHandleType__enumvalues = CUexternalMemoryHandleType_enum__enumvalues
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(Structure):
    pass

class union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle(Union):
    pass

class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle._pack_ = 1 # source:False
union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32),
    ('nvSciBufObject', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._fields_ = [
    ('type', CUexternalMemoryHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('arrayDesc', CUDA_ARRAY3D_DESCRIPTOR),
    ('numLevels', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st

# values for enumeration 'CUexternalSemaphoreHandleType_enum'
CUexternalSemaphoreHandleType_enum__enumvalues = {
    1: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD',
    2: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32',
    3: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    4: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE',
    5: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE',
    6: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC',
    7: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX',
    8: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT',
    9: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD',
    10: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32',
}
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
CUexternalSemaphoreHandleType_enum = ctypes.c_uint32 # enum
CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum
CUexternalSemaphoreHandleType__enumvalues = CUexternalSemaphoreHandleType_enum__enumvalues
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(Structure):
    pass

class union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle(Union):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32),
    ('nvSciSyncObj', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._fields_ = [
    ('type', CUexternalSemaphoreHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync(Union):
    pass

union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
]

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params._fields_ = [
    ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence),
    ('nvSciSync', union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync),
    ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 12),
]

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._fields_ = [
    ('params', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync(Union):
    pass

union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
    ('timeoutMs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params._fields_ = [
    ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence),
    ('nvSciSync', union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync),
    ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 10),
]

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._fields_ = [
    ('params', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
class struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st))),
    ('paramsArray', ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
class struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st))),
    ('paramsArray', ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUDA_EXT_SEM_WAIT_NODE_PARAMS = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUmemGenericAllocationHandle_v1 = ctypes.c_uint64
CUmemGenericAllocationHandle = ctypes.c_uint64

# values for enumeration 'CUmemAllocationHandleType_enum'
CUmemAllocationHandleType_enum__enumvalues = {
    0: 'CU_MEM_HANDLE_TYPE_NONE',
    1: 'CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR',
    2: 'CU_MEM_HANDLE_TYPE_WIN32',
    4: 'CU_MEM_HANDLE_TYPE_WIN32_KMT',
    2147483647: 'CU_MEM_HANDLE_TYPE_MAX',
}
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_HANDLE_TYPE_WIN32 = 2
CU_MEM_HANDLE_TYPE_WIN32_KMT = 4
CU_MEM_HANDLE_TYPE_MAX = 2147483647
CUmemAllocationHandleType_enum = ctypes.c_uint32 # enum
CUmemAllocationHandleType = CUmemAllocationHandleType_enum
CUmemAllocationHandleType__enumvalues = CUmemAllocationHandleType_enum__enumvalues

# values for enumeration 'CUmemAccess_flags_enum'
CUmemAccess_flags_enum__enumvalues = {
    0: 'CU_MEM_ACCESS_FLAGS_PROT_NONE',
    1: 'CU_MEM_ACCESS_FLAGS_PROT_READ',
    3: 'CU_MEM_ACCESS_FLAGS_PROT_READWRITE',
    2147483647: 'CU_MEM_ACCESS_FLAGS_PROT_MAX',
}
CU_MEM_ACCESS_FLAGS_PROT_NONE = 0
CU_MEM_ACCESS_FLAGS_PROT_READ = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647
CUmemAccess_flags_enum = ctypes.c_uint32 # enum
CUmemAccess_flags = CUmemAccess_flags_enum
CUmemAccess_flags__enumvalues = CUmemAccess_flags_enum__enumvalues

# values for enumeration 'CUmemLocationType_enum'
CUmemLocationType_enum__enumvalues = {
    0: 'CU_MEM_LOCATION_TYPE_INVALID',
    1: 'CU_MEM_LOCATION_TYPE_DEVICE',
    2147483647: 'CU_MEM_LOCATION_TYPE_MAX',
}
CU_MEM_LOCATION_TYPE_INVALID = 0
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_LOCATION_TYPE_MAX = 2147483647
CUmemLocationType_enum = ctypes.c_uint32 # enum
CUmemLocationType = CUmemLocationType_enum
CUmemLocationType__enumvalues = CUmemLocationType_enum__enumvalues

# values for enumeration 'CUmemAllocationType_enum'
CUmemAllocationType_enum__enumvalues = {
    0: 'CU_MEM_ALLOCATION_TYPE_INVALID',
    1: 'CU_MEM_ALLOCATION_TYPE_PINNED',
    2147483647: 'CU_MEM_ALLOCATION_TYPE_MAX',
}
CU_MEM_ALLOCATION_TYPE_INVALID = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ALLOCATION_TYPE_MAX = 2147483647
CUmemAllocationType_enum = ctypes.c_uint32 # enum
CUmemAllocationType = CUmemAllocationType_enum
CUmemAllocationType__enumvalues = CUmemAllocationType_enum__enumvalues

# values for enumeration 'CUmemAllocationGranularity_flags_enum'
CUmemAllocationGranularity_flags_enum__enumvalues = {
    0: 'CU_MEM_ALLOC_GRANULARITY_MINIMUM',
    1: 'CU_MEM_ALLOC_GRANULARITY_RECOMMENDED',
}
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
CUmemAllocationGranularity_flags_enum = ctypes.c_uint32 # enum
CUmemAllocationGranularity_flags = CUmemAllocationGranularity_flags_enum
CUmemAllocationGranularity_flags__enumvalues = CUmemAllocationGranularity_flags_enum__enumvalues

# values for enumeration 'CUarraySparseSubresourceType_enum'
CUarraySparseSubresourceType_enum__enumvalues = {
    0: 'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL',
    1: 'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL',
}
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
CUarraySparseSubresourceType_enum = ctypes.c_uint32 # enum
CUarraySparseSubresourceType = CUarraySparseSubresourceType_enum
CUarraySparseSubresourceType__enumvalues = CUarraySparseSubresourceType_enum__enumvalues

# values for enumeration 'CUmemOperationType_enum'
CUmemOperationType_enum__enumvalues = {
    1: 'CU_MEM_OPERATION_TYPE_MAP',
    2: 'CU_MEM_OPERATION_TYPE_UNMAP',
}
CU_MEM_OPERATION_TYPE_MAP = 1
CU_MEM_OPERATION_TYPE_UNMAP = 2
CUmemOperationType_enum = ctypes.c_uint32 # enum
CUmemOperationType = CUmemOperationType_enum
CUmemOperationType__enumvalues = CUmemOperationType_enum__enumvalues

# values for enumeration 'CUmemHandleType_enum'
CUmemHandleType_enum__enumvalues = {
    0: 'CU_MEM_HANDLE_TYPE_GENERIC',
}
CU_MEM_HANDLE_TYPE_GENERIC = 0
CUmemHandleType_enum = ctypes.c_uint32 # enum
CUmemHandleType = CUmemHandleType_enum
CUmemHandleType__enumvalues = CUmemHandleType_enum__enumvalues
class struct_CUarrayMapInfo_st(Structure):
    pass

class union_CUarrayMapInfo_st_resource(Union):
    pass

union_CUarrayMapInfo_st_resource._pack_ = 1 # source:False
union_CUarrayMapInfo_st_resource._fields_ = [
    ('mipmap', ctypes.POINTER(struct_CUmipmappedArray_st)),
    ('array', ctypes.POINTER(struct_CUarray_st)),
]

class union_CUarrayMapInfo_st_subresource(Union):
    pass

class struct_CUarrayMapInfo_st_1_sparseLevel(Structure):
    pass

struct_CUarrayMapInfo_st_1_sparseLevel._pack_ = 1 # source:False
struct_CUarrayMapInfo_st_1_sparseLevel._fields_ = [
    ('level', ctypes.c_uint32),
    ('layer', ctypes.c_uint32),
    ('offsetX', ctypes.c_uint32),
    ('offsetY', ctypes.c_uint32),
    ('offsetZ', ctypes.c_uint32),
    ('extentWidth', ctypes.c_uint32),
    ('extentHeight', ctypes.c_uint32),
    ('extentDepth', ctypes.c_uint32),
]

class struct_CUarrayMapInfo_st_1_miptail(Structure):
    pass

struct_CUarrayMapInfo_st_1_miptail._pack_ = 1 # source:False
struct_CUarrayMapInfo_st_1_miptail._fields_ = [
    ('layer', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

union_CUarrayMapInfo_st_subresource._pack_ = 1 # source:False
union_CUarrayMapInfo_st_subresource._fields_ = [
    ('sparseLevel', struct_CUarrayMapInfo_st_1_sparseLevel),
    ('miptail', struct_CUarrayMapInfo_st_1_miptail),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

class union_CUarrayMapInfo_st_memHandle(Union):
    pass

union_CUarrayMapInfo_st_memHandle._pack_ = 1 # source:False
union_CUarrayMapInfo_st_memHandle._fields_ = [
    ('memHandle', ctypes.c_uint64),
]

struct_CUarrayMapInfo_st._pack_ = 1 # source:False
struct_CUarrayMapInfo_st._fields_ = [
    ('resourceType', CUresourcetype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('resource', union_CUarrayMapInfo_st_resource),
    ('subresourceType', CUarraySparseSubresourceType),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('subresource', union_CUarrayMapInfo_st_subresource),
    ('memOperationType', CUmemOperationType),
    ('memHandleType', CUmemHandleType),
    ('memHandle', union_CUarrayMapInfo_st_memHandle),
    ('offset', ctypes.c_uint64),
    ('deviceBitMask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 2),
]

CUarrayMapInfo_v1 = struct_CUarrayMapInfo_st
CUarrayMapInfo = struct_CUarrayMapInfo_st
class struct_CUmemLocation_st(Structure):
    pass

struct_CUmemLocation_st._pack_ = 1 # source:False
struct_CUmemLocation_st._fields_ = [
    ('type', CUmemLocationType),
    ('id', ctypes.c_int32),
]

CUmemLocation_v1 = struct_CUmemLocation_st
CUmemLocation = struct_CUmemLocation_st

# values for enumeration 'CUmemAllocationCompType_enum'
CUmemAllocationCompType_enum__enumvalues = {
    0: 'CU_MEM_ALLOCATION_COMP_NONE',
    1: 'CU_MEM_ALLOCATION_COMP_GENERIC',
}
CU_MEM_ALLOCATION_COMP_NONE = 0
CU_MEM_ALLOCATION_COMP_GENERIC = 1
CUmemAllocationCompType_enum = ctypes.c_uint32 # enum
CUmemAllocationCompType = CUmemAllocationCompType_enum
CUmemAllocationCompType__enumvalues = CUmemAllocationCompType_enum__enumvalues
class struct_CUmemAllocationProp_st(Structure):
    pass

class struct_CUmemAllocationProp_st_allocFlags(Structure):
    pass

struct_CUmemAllocationProp_st_allocFlags._pack_ = 1 # source:False
struct_CUmemAllocationProp_st_allocFlags._fields_ = [
    ('compressionType', ctypes.c_ubyte),
    ('gpuDirectRDMACapable', ctypes.c_ubyte),
    ('usage', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 4),
]

struct_CUmemAllocationProp_st._pack_ = 1 # source:False
struct_CUmemAllocationProp_st._fields_ = [
    ('type', CUmemAllocationType),
    ('requestedHandleTypes', CUmemAllocationHandleType),
    ('location', CUmemLocation),
    ('win32HandleMetaData', ctypes.POINTER(None)),
    ('allocFlags', struct_CUmemAllocationProp_st_allocFlags),
]

CUmemAllocationProp_v1 = struct_CUmemAllocationProp_st
CUmemAllocationProp = struct_CUmemAllocationProp_st
class struct_CUmemAccessDesc_st(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('location', CUmemLocation),
    ('flags', CUmemAccess_flags),
     ]

CUmemAccessDesc_v1 = struct_CUmemAccessDesc_st
CUmemAccessDesc = struct_CUmemAccessDesc_st

# values for enumeration 'CUgraphExecUpdateResult_enum'
CUgraphExecUpdateResult_enum__enumvalues = {
    0: 'CU_GRAPH_EXEC_UPDATE_SUCCESS',
    1: 'CU_GRAPH_EXEC_UPDATE_ERROR',
    2: 'CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED',
    3: 'CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED',
    4: 'CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED',
    5: 'CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED',
    6: 'CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED',
    7: 'CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE',
}
CU_GRAPH_EXEC_UPDATE_SUCCESS = 0
CU_GRAPH_EXEC_UPDATE_ERROR = 1
CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2
CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3
CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4
CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5
CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6
CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7
CUgraphExecUpdateResult_enum = ctypes.c_uint32 # enum
CUgraphExecUpdateResult = CUgraphExecUpdateResult_enum
CUgraphExecUpdateResult__enumvalues = CUgraphExecUpdateResult_enum__enumvalues

# values for enumeration 'CUmemPool_attribute_enum'
CUmemPool_attribute_enum__enumvalues = {
    1: 'CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES',
    2: 'CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC',
    3: 'CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES',
    4: 'CU_MEMPOOL_ATTR_RELEASE_THRESHOLD',
    5: 'CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT',
    6: 'CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH',
    7: 'CU_MEMPOOL_ATTR_USED_MEM_CURRENT',
    8: 'CU_MEMPOOL_ATTR_USED_MEM_HIGH',
}
CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1
CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2
CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5
CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6
CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7
CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
CUmemPool_attribute_enum = ctypes.c_uint32 # enum
CUmemPool_attribute = CUmemPool_attribute_enum
CUmemPool_attribute__enumvalues = CUmemPool_attribute_enum__enumvalues
class struct_CUmemPoolProps_st(Structure):
    pass

struct_CUmemPoolProps_st._pack_ = 1 # source:False
struct_CUmemPoolProps_st._fields_ = [
    ('allocType', CUmemAllocationType),
    ('handleTypes', CUmemAllocationHandleType),
    ('location', CUmemLocation),
    ('win32SecurityAttributes', ctypes.POINTER(None)),
    ('reserved', ctypes.c_ubyte * 64),
]

CUmemPoolProps_v1 = struct_CUmemPoolProps_st
CUmemPoolProps = struct_CUmemPoolProps_st
class struct_CUmemPoolPtrExportData_st(Structure):
    pass

struct_CUmemPoolPtrExportData_st._pack_ = 1 # source:False
struct_CUmemPoolPtrExportData_st._fields_ = [
    ('reserved', ctypes.c_ubyte * 64),
]

CUmemPoolPtrExportData_v1 = struct_CUmemPoolPtrExportData_st
CUmemPoolPtrExportData = struct_CUmemPoolPtrExportData_st
class struct_CUDA_MEM_ALLOC_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._fields_ = [
    ('poolProps', CUmemPoolProps),
    ('accessDescs', ctypes.POINTER(struct_CUmemAccessDesc_st)),
    ('accessDescCount', ctypes.c_uint64),
    ('bytesize', ctypes.c_uint64),
    ('dptr', ctypes.c_uint64),
]

CUDA_MEM_ALLOC_NODE_PARAMS = struct_CUDA_MEM_ALLOC_NODE_PARAMS_st

# values for enumeration 'CUgraphMem_attribute_enum'
CUgraphMem_attribute_enum__enumvalues = {
    0: 'CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT',
    1: 'CU_GRAPH_MEM_ATTR_USED_MEM_HIGH',
    2: 'CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT',
    3: 'CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH',
}
CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3
CUgraphMem_attribute_enum = ctypes.c_uint32 # enum
CUgraphMem_attribute = CUgraphMem_attribute_enum
CUgraphMem_attribute__enumvalues = CUgraphMem_attribute_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesOptions_enum'
CUflushGPUDirectRDMAWritesOptions_enum__enumvalues = {
    1: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST',
    2: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2
CUflushGPUDirectRDMAWritesOptions_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesOptions = CUflushGPUDirectRDMAWritesOptions_enum
CUflushGPUDirectRDMAWritesOptions__enumvalues = CUflushGPUDirectRDMAWritesOptions_enum__enumvalues

# values for enumeration 'CUGPUDirectRDMAWritesOrdering_enum'
CUGPUDirectRDMAWritesOrdering_enum__enumvalues = {
    0: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE',
    100: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER',
    200: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES',
}
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200
CUGPUDirectRDMAWritesOrdering_enum = ctypes.c_uint32 # enum
CUGPUDirectRDMAWritesOrdering = CUGPUDirectRDMAWritesOrdering_enum
CUGPUDirectRDMAWritesOrdering__enumvalues = CUGPUDirectRDMAWritesOrdering_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesScope_enum'
CUflushGPUDirectRDMAWritesScope_enum__enumvalues = {
    100: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER',
    200: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200
CUflushGPUDirectRDMAWritesScope_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesScope = CUflushGPUDirectRDMAWritesScope_enum
CUflushGPUDirectRDMAWritesScope__enumvalues = CUflushGPUDirectRDMAWritesScope_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesTarget_enum'
CUflushGPUDirectRDMAWritesTarget_enum__enumvalues = {
    0: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0
CUflushGPUDirectRDMAWritesTarget_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesTarget = CUflushGPUDirectRDMAWritesTarget_enum
CUflushGPUDirectRDMAWritesTarget__enumvalues = CUflushGPUDirectRDMAWritesTarget_enum__enumvalues

# values for enumeration 'CUgraphDebugDot_flags_enum'
CUgraphDebugDot_flags_enum__enumvalues = {
    1: 'CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE',
    2: 'CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES',
    4: 'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS',
    8: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS',
    16: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS',
    32: 'CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS',
    64: 'CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS',
    128: 'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS',
    256: 'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS',
    512: 'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES',
    1024: 'CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES',
    2048: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS',
    4096: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS',
}
CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1
CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4
CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8
CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16
CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32
CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512
CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096
CUgraphDebugDot_flags_enum = ctypes.c_uint32 # enum
CUgraphDebugDot_flags = CUgraphDebugDot_flags_enum
CUgraphDebugDot_flags__enumvalues = CUgraphDebugDot_flags_enum__enumvalues

# values for enumeration 'CUuserObject_flags_enum'
CUuserObject_flags_enum__enumvalues = {
    1: 'CU_USER_OBJECT_NO_DESTRUCTOR_SYNC',
}
CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1
CUuserObject_flags_enum = ctypes.c_uint32 # enum
CUuserObject_flags = CUuserObject_flags_enum
CUuserObject_flags__enumvalues = CUuserObject_flags_enum__enumvalues

# values for enumeration 'CUuserObjectRetain_flags_enum'
CUuserObjectRetain_flags_enum__enumvalues = {
    1: 'CU_GRAPH_USER_OBJECT_MOVE',
}
CU_GRAPH_USER_OBJECT_MOVE = 1
CUuserObjectRetain_flags_enum = ctypes.c_uint32 # enum
CUuserObjectRetain_flags = CUuserObjectRetain_flags_enum
CUuserObjectRetain_flags__enumvalues = CUuserObjectRetain_flags_enum__enumvalues

# values for enumeration 'CUgraphInstantiate_flags_enum'
CUgraphInstantiate_flags_enum__enumvalues = {
    1: 'CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH',
}
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1
CUgraphInstantiate_flags_enum = ctypes.c_uint32 # enum
CUgraphInstantiate_flags = CUgraphInstantiate_flags_enum
CUgraphInstantiate_flags__enumvalues = CUgraphInstantiate_flags_enum__enumvalues
try:
    cuGetErrorString = _libraries['libcuda.so'].cuGetErrorString
    cuGetErrorString.restype = CUresult
    cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    cuGetErrorName = _libraries['libcuda.so'].cuGetErrorName
    cuGetErrorName.restype = CUresult
    cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    cuInit = _libraries['libcuda.so'].cuInit
    cuInit.restype = CUresult
    cuInit.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDriverGetVersion = _libraries['libcuda.so'].cuDriverGetVersion
    cuDriverGetVersion.restype = CUresult
    cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDeviceGet = _libraries['libcuda.so'].cuDeviceGet
    cuDeviceGet.restype = CUresult
    cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
except AttributeError:
    pass
try:
    cuDeviceGetCount = _libraries['libcuda.so'].cuDeviceGetCount
    cuDeviceGetCount.restype = CUresult
    cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDeviceGetName = _libraries['libcuda.so'].cuDeviceGetName
    cuDeviceGetName.restype = CUresult
    cuDeviceGetName.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetUuid = _libraries['libcuda.so'].cuDeviceGetUuid
    cuDeviceGetUuid.restype = CUresult
    cuDeviceGetUuid.argtypes = [ctypes.POINTER(struct_CUuuid_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetUuid_v2 = _libraries['libcuda.so'].cuDeviceGetUuid_v2
    cuDeviceGetUuid_v2.restype = CUresult
    cuDeviceGetUuid_v2.argtypes = [ctypes.POINTER(struct_CUuuid_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetLuid = _libraries['libcuda.so'].cuDeviceGetLuid
    cuDeviceGetLuid.restype = CUresult
    cuDeviceGetLuid.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceTotalMem_v2 = _libraries['libcuda.so'].cuDeviceTotalMem_v2
    cuDeviceTotalMem_v2.restype = CUresult
    cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetTexture1DLinearMaxWidth = _libraries['libcuda.so'].cuDeviceGetTexture1DLinearMaxWidth
    cuDeviceGetTexture1DLinearMaxWidth.restype = CUresult
    cuDeviceGetTexture1DLinearMaxWidth.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUarray_format, ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetAttribute = _libraries['libcuda.so'].cuDeviceGetAttribute
    cuDeviceGetAttribute.restype = CUresult
    cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice_attribute, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetNvSciSyncAttributes = _libraries['libcuda.so'].cuDeviceGetNvSciSyncAttributes
    cuDeviceGetNvSciSyncAttributes.restype = CUresult
    cuDeviceGetNvSciSyncAttributes.argtypes = [ctypes.POINTER(None), CUdevice, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuDeviceSetMemPool = _libraries['libcuda.so'].cuDeviceSetMemPool
    cuDeviceSetMemPool.restype = CUresult
    cuDeviceSetMemPool.argtypes = [CUdevice, CUmemoryPool]
except AttributeError:
    pass
try:
    cuDeviceGetMemPool = _libraries['libcuda.so'].cuDeviceGetMemPool
    cuDeviceGetMemPool.restype = CUresult
    cuDeviceGetMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetDefaultMemPool = _libraries['libcuda.so'].cuDeviceGetDefaultMemPool
    cuDeviceGetDefaultMemPool.restype = CUresult
    cuDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), CUdevice]
except AttributeError:
    pass
try:
    cuFlushGPUDirectRDMAWrites = _libraries['libcuda.so'].cuFlushGPUDirectRDMAWrites
    cuFlushGPUDirectRDMAWrites.restype = CUresult
    cuFlushGPUDirectRDMAWrites.argtypes = [CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope]
except AttributeError:
    pass
try:
    cuDeviceGetProperties = _libraries['libcuda.so'].cuDeviceGetProperties
    cuDeviceGetProperties.restype = CUresult
    cuDeviceGetProperties.argtypes = [ctypes.POINTER(struct_CUdevprop_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceComputeCapability = _libraries['libcuda.so'].cuDeviceComputeCapability
    cuDeviceComputeCapability.restype = CUresult
    cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxRetain = _libraries['libcuda.so'].cuDevicePrimaryCtxRetain
    cuDevicePrimaryCtxRetain.restype = CUresult
    cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxRelease_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxRelease_v2
    cuDevicePrimaryCtxRelease_v2.restype = CUresult
    cuDevicePrimaryCtxRelease_v2.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxSetFlags_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxSetFlags_v2
    cuDevicePrimaryCtxSetFlags_v2.restype = CUresult
    cuDevicePrimaryCtxSetFlags_v2.argtypes = [CUdevice, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxGetState = _libraries['libcuda.so'].cuDevicePrimaryCtxGetState
    cuDevicePrimaryCtxGetState.restype = CUresult
    cuDevicePrimaryCtxGetState.argtypes = [CUdevice, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxReset_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxReset_v2
    cuDevicePrimaryCtxReset_v2.restype = CUresult
    cuDevicePrimaryCtxReset_v2.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetExecAffinitySupport = _libraries['libcuda.so'].cuDeviceGetExecAffinitySupport
    cuDeviceGetExecAffinitySupport.restype = CUresult
    cuDeviceGetExecAffinitySupport.argtypes = [ctypes.POINTER(ctypes.c_int32), CUexecAffinityType, CUdevice]
except AttributeError:
    pass
try:
    cuCtxCreate_v2 = _libraries['libcuda.so'].cuCtxCreate_v2
    cuCtxCreate_v2.restype = CUresult
    cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuCtxCreate_v3 = _libraries['libcuda.so'].cuCtxCreate_v3
    cuCtxCreate_v3.restype = CUresult
    cuCtxCreate_v3.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.POINTER(struct_CUexecAffinityParam_st), ctypes.c_int32, ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuCtxDestroy_v2 = _libraries['libcuda.so'].cuCtxDestroy_v2
    cuCtxDestroy_v2.restype = CUresult
    cuCtxDestroy_v2.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxPushCurrent_v2 = _libraries['libcuda.so'].cuCtxPushCurrent_v2
    cuCtxPushCurrent_v2.restype = CUresult
    cuCtxPushCurrent_v2.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxPopCurrent_v2 = _libraries['libcuda.so'].cuCtxPopCurrent_v2
    cuCtxPopCurrent_v2.restype = CUresult
    cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuCtxSetCurrent = _libraries['libcuda.so'].cuCtxSetCurrent
    cuCtxSetCurrent.restype = CUresult
    cuCtxSetCurrent.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxGetCurrent = _libraries['libcuda.so'].cuCtxGetCurrent
    cuCtxGetCurrent.restype = CUresult
    cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuCtxGetDevice = _libraries['libcuda.so'].cuCtxGetDevice
    cuCtxGetDevice.restype = CUresult
    cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuCtxGetFlags = _libraries['libcuda.so'].cuCtxGetFlags
    cuCtxGetFlags.restype = CUresult
    cuCtxGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuCtxSynchronize = _libraries['libcuda.so'].cuCtxSynchronize
    cuCtxSynchronize.restype = CUresult
    cuCtxSynchronize.argtypes = []
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    cuCtxSetLimit = _libraries['libcuda.so'].cuCtxSetLimit
    cuCtxSetLimit.restype = CUresult
    cuCtxSetLimit.argtypes = [CUlimit, size_t]
except AttributeError:
    pass
try:
    cuCtxGetLimit = _libraries['libcuda.so'].cuCtxGetLimit
    cuCtxGetLimit.restype = CUresult
    cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUlimit]
except AttributeError:
    pass
try:
    cuCtxGetCacheConfig = _libraries['libcuda.so'].cuCtxGetCacheConfig
    cuCtxGetCacheConfig.restype = CUresult
    cuCtxGetCacheConfig.argtypes = [ctypes.POINTER(CUfunc_cache_enum)]
except AttributeError:
    pass
try:
    cuCtxSetCacheConfig = _libraries['libcuda.so'].cuCtxSetCacheConfig
    cuCtxSetCacheConfig.restype = CUresult
    cuCtxSetCacheConfig.argtypes = [CUfunc_cache]
except AttributeError:
    pass
try:
    cuCtxGetSharedMemConfig = _libraries['libcuda.so'].cuCtxGetSharedMemConfig
    cuCtxGetSharedMemConfig.restype = CUresult
    cuCtxGetSharedMemConfig.argtypes = [ctypes.POINTER(CUsharedconfig_enum)]
except AttributeError:
    pass
try:
    cuCtxSetSharedMemConfig = _libraries['libcuda.so'].cuCtxSetSharedMemConfig
    cuCtxSetSharedMemConfig.restype = CUresult
    cuCtxSetSharedMemConfig.argtypes = [CUsharedconfig]
except AttributeError:
    pass
try:
    cuCtxGetApiVersion = _libraries['libcuda.so'].cuCtxGetApiVersion
    cuCtxGetApiVersion.restype = CUresult
    cuCtxGetApiVersion.argtypes = [CUcontext, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuCtxGetStreamPriorityRange = _libraries['libcuda.so'].cuCtxGetStreamPriorityRange
    cuCtxGetStreamPriorityRange.restype = CUresult
    cuCtxGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuCtxResetPersistingL2Cache = _libraries['libcuda.so'].cuCtxResetPersistingL2Cache
    cuCtxResetPersistingL2Cache.restype = CUresult
    cuCtxResetPersistingL2Cache.argtypes = []
except AttributeError:
    pass
try:
    cuCtxGetExecAffinity = _libraries['libcuda.so'].cuCtxGetExecAffinity
    cuCtxGetExecAffinity.restype = CUresult
    cuCtxGetExecAffinity.argtypes = [ctypes.POINTER(struct_CUexecAffinityParam_st), CUexecAffinityType]
except AttributeError:
    pass
try:
    cuCtxAttach = _libraries['libcuda.so'].cuCtxAttach
    cuCtxAttach.restype = CUresult
    cuCtxAttach.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuCtxDetach = _libraries['libcuda.so'].cuCtxDetach
    cuCtxDetach.restype = CUresult
    cuCtxDetach.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuModuleLoad = _libraries['libcuda.so'].cuModuleLoad
    cuModuleLoad.restype = CUresult
    cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleLoadData = _libraries['libcuda.so'].cuModuleLoadData
    cuModuleLoadData.restype = CUresult
    cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuModuleLoadDataEx = _libraries['libcuda.so'].cuModuleLoadDataEx
    cuModuleLoadDataEx.restype = CUresult
    cuModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuModuleLoadFatBinary = _libraries['libcuda.so'].cuModuleLoadFatBinary
    cuModuleLoadFatBinary.restype = CUresult
    cuModuleLoadFatBinary.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuModuleUnload = _libraries['libcuda.so'].cuModuleUnload
    cuModuleUnload.restype = CUresult
    cuModuleUnload.argtypes = [CUmodule]
except AttributeError:
    pass
try:
    cuModuleGetFunction = _libraries['libcuda.so'].cuModuleGetFunction
    cuModuleGetFunction.restype = CUresult
    cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUfunc_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetGlobal_v2 = _libraries['libcuda.so'].cuModuleGetGlobal_v2
    cuModuleGetGlobal_v2.restype = CUresult
    cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetTexRef = _libraries['libcuda.so'].cuModuleGetTexRef
    cuModuleGetTexRef.restype = CUresult
    cuModuleGetTexRef.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUtexref_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetSurfRef = _libraries['libcuda.so'].cuModuleGetSurfRef
    cuModuleGetSurfRef.restype = CUresult
    cuModuleGetSurfRef.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUsurfref_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuLinkCreate_v2 = _libraries['libcuda.so'].cuLinkCreate_v2
    cuLinkCreate_v2.restype = CUresult
    cuLinkCreate_v2.argtypes = [ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(struct_CUlinkState_st))]
except AttributeError:
    pass
try:
    cuLinkAddData_v2 = _libraries['libcuda.so'].cuLinkAddData_v2
    cuLinkAddData_v2.restype = CUresult
    cuLinkAddData_v2.argtypes = [CUlinkState, CUjitInputType, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLinkAddFile_v2 = _libraries['libcuda.so'].cuLinkAddFile_v2
    cuLinkAddFile_v2.restype = CUresult
    cuLinkAddFile_v2.argtypes = [CUlinkState, CUjitInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLinkComplete = _libraries['libcuda.so'].cuLinkComplete
    cuLinkComplete.restype = CUresult
    cuLinkComplete.argtypes = [CUlinkState, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuLinkDestroy = _libraries['libcuda.so'].cuLinkDestroy
    cuLinkDestroy.restype = CUresult
    cuLinkDestroy.argtypes = [CUlinkState]
except AttributeError:
    pass
try:
    cuMemGetInfo_v2 = _libraries['libcuda.so'].cuMemGetInfo_v2
    cuMemGetInfo_v2.restype = CUresult
    cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuMemAlloc_v2 = _libraries['libcuda.so'].cuMemAlloc_v2
    cuMemAlloc_v2.restype = CUresult
    cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t]
except AttributeError:
    pass
try:
    cuMemAllocPitch_v2 = _libraries['libcuda.so'].cuMemAllocPitch_v2
    cuMemAllocPitch_v2.restype = CUresult
    cuMemAllocPitch_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemFree_v2 = _libraries['libcuda.so'].cuMemFree_v2
    cuMemFree_v2.restype = CUresult
    cuMemFree_v2.argtypes = [CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemGetAddressRange_v2 = _libraries['libcuda.so'].cuMemGetAddressRange_v2
    cuMemGetAddressRange_v2.restype = CUresult
    cuMemGetAddressRange_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemAllocHost_v2 = _libraries['libcuda.so'].cuMemAllocHost_v2
    cuMemAllocHost_v2.restype = CUresult
    cuMemAllocHost_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t]
except AttributeError:
    pass
try:
    cuMemFreeHost = _libraries['libcuda.so'].cuMemFreeHost
    cuMemFreeHost.restype = CUresult
    cuMemFreeHost.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemHostAlloc = _libraries['libcuda.so'].cuMemHostAlloc
    cuMemHostAlloc.restype = CUresult
    cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostGetDevicePointer_v2 = _libraries['libcuda.so'].cuMemHostGetDevicePointer_v2
    cuMemHostGetDevicePointer_v2.restype = CUresult
    cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostGetFlags = _libraries['libcuda.so'].cuMemHostGetFlags
    cuMemHostGetFlags.restype = CUresult
    cuMemHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemAllocManaged = _libraries['libcuda.so'].cuMemAllocManaged
    cuMemAllocManaged.restype = CUresult
    cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDeviceGetByPCIBusId = _libraries['libcuda.so'].cuDeviceGetByPCIBusId
    cuDeviceGetByPCIBusId.restype = CUresult
    cuDeviceGetByPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuDeviceGetPCIBusId = _libraries['libcuda.so'].cuDeviceGetPCIBusId
    cuDeviceGetPCIBusId.restype = CUresult
    cuDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, CUdevice]
except AttributeError:
    pass
try:
    cuIpcGetEventHandle = _libraries['libcuda.so'].cuIpcGetEventHandle
    cuIpcGetEventHandle.restype = CUresult
    cuIpcGetEventHandle.argtypes = [ctypes.POINTER(struct_CUipcEventHandle_st), CUevent]
except AttributeError:
    pass
try:
    cuIpcOpenEventHandle = _libraries['libcuda.so'].cuIpcOpenEventHandle
    cuIpcOpenEventHandle.restype = CUresult
    cuIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUevent_st)), CUipcEventHandle]
except AttributeError:
    pass
try:
    cuIpcGetMemHandle = _libraries['libcuda.so'].cuIpcGetMemHandle
    cuIpcGetMemHandle.restype = CUresult
    cuIpcGetMemHandle.argtypes = [ctypes.POINTER(struct_CUipcMemHandle_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuIpcOpenMemHandle_v2 = _libraries['libcuda.so'].cuIpcOpenMemHandle_v2
    cuIpcOpenMemHandle_v2.restype = CUresult
    cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUipcMemHandle, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuIpcCloseMemHandle = _libraries['libcuda.so'].cuIpcCloseMemHandle
    cuIpcCloseMemHandle.restype = CUresult
    cuIpcCloseMemHandle.argtypes = [CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemHostRegister_v2 = _libraries['libcuda.so'].cuMemHostRegister_v2
    cuMemHostRegister_v2.restype = CUresult
    cuMemHostRegister_v2.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostUnregister = _libraries['libcuda.so'].cuMemHostUnregister
    cuMemHostUnregister.restype = CUresult
    cuMemHostUnregister.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemcpy = _libraries['libcuda.so'].cuMemcpy
    cuMemcpy.restype = CUresult
    cuMemcpy.argtypes = [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyPeer = _libraries['libcuda.so'].cuMemcpyPeer
    cuMemcpyPeer.restype = CUresult
    cuMemcpyPeer.argtypes = [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t]
except AttributeError:
    pass
try:
    cuMemcpyHtoD_v2 = _libraries['libcuda.so'].cuMemcpyHtoD_v2
    cuMemcpyHtoD_v2.restype = CUresult
    cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoH_v2 = _libraries['libcuda.so'].cuMemcpyDtoH_v2
    cuMemcpyDtoH_v2.restype = CUresult
    cuMemcpyDtoH_v2.argtypes = [ctypes.POINTER(None), CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoD_v2 = _libraries['libcuda.so'].cuMemcpyDtoD_v2
    cuMemcpyDtoD_v2.restype = CUresult
    cuMemcpyDtoD_v2.argtypes = [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoA_v2 = _libraries['libcuda.so'].cuMemcpyDtoA_v2
    cuMemcpyDtoA_v2.restype = CUresult
    cuMemcpyDtoA_v2.argtypes = [CUarray, size_t, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoD_v2 = _libraries['libcuda.so'].cuMemcpyAtoD_v2
    cuMemcpyAtoD_v2.restype = CUresult
    cuMemcpyAtoD_v2.argtypes = [CUdeviceptr, CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpyHtoA_v2 = _libraries['libcuda.so'].cuMemcpyHtoA_v2
    cuMemcpyHtoA_v2.restype = CUresult
    cuMemcpyHtoA_v2.argtypes = [CUarray, size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoH_v2 = _libraries['libcuda.so'].cuMemcpyAtoH_v2
    cuMemcpyAtoH_v2.restype = CUresult
    cuMemcpyAtoH_v2.argtypes = [ctypes.POINTER(None), CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoA_v2 = _libraries['libcuda.so'].cuMemcpyAtoA_v2
    cuMemcpyAtoA_v2.restype = CUresult
    cuMemcpyAtoA_v2.argtypes = [CUarray, size_t, CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpy2D_v2 = _libraries['libcuda.so'].cuMemcpy2D_v2
    cuMemcpy2D_v2.restype = CUresult
    cuMemcpy2D_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st)]
except AttributeError:
    pass
try:
    cuMemcpy2DUnaligned_v2 = _libraries['libcuda.so'].cuMemcpy2DUnaligned_v2
    cuMemcpy2DUnaligned_v2.restype = CUresult
    cuMemcpy2DUnaligned_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st)]
except AttributeError:
    pass
try:
    cuMemcpy3D_v2 = _libraries['libcuda.so'].cuMemcpy3D_v2
    cuMemcpy3D_v2.restype = CUresult
    cuMemcpy3D_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuMemcpy3DPeer = _libraries['libcuda.so'].cuMemcpy3DPeer
    cuMemcpy3DPeer.restype = CUresult
    cuMemcpy3DPeer.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_PEER_st)]
except AttributeError:
    pass
try:
    cuMemcpyAsync = _libraries['libcuda.so'].cuMemcpyAsync
    cuMemcpyAsync.restype = CUresult
    cuMemcpyAsync.argtypes = [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyPeerAsync = _libraries['libcuda.so'].cuMemcpyPeerAsync
    cuMemcpyPeerAsync.restype = CUresult
    cuMemcpyPeerAsync.argtypes = [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyHtoDAsync_v2 = _libraries['libcuda.so'].cuMemcpyHtoDAsync_v2
    cuMemcpyHtoDAsync_v2.restype = CUresult
    cuMemcpyHtoDAsync_v2.argtypes = [CUdeviceptr, ctypes.POINTER(None), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyDtoHAsync_v2 = _libraries['libcuda.so'].cuMemcpyDtoHAsync_v2
    cuMemcpyDtoHAsync_v2.restype = CUresult
    cuMemcpyDtoHAsync_v2.argtypes = [ctypes.POINTER(None), CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyDtoDAsync_v2 = _libraries['libcuda.so'].cuMemcpyDtoDAsync_v2
    cuMemcpyDtoDAsync_v2.restype = CUresult
    cuMemcpyDtoDAsync_v2.argtypes = [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyHtoAAsync_v2 = _libraries['libcuda.so'].cuMemcpyHtoAAsync_v2
    cuMemcpyHtoAAsync_v2.restype = CUresult
    cuMemcpyHtoAAsync_v2.argtypes = [CUarray, size_t, ctypes.POINTER(None), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyAtoHAsync_v2 = _libraries['libcuda.so'].cuMemcpyAtoHAsync_v2
    cuMemcpyAtoHAsync_v2.restype = CUresult
    cuMemcpyAtoHAsync_v2.argtypes = [ctypes.POINTER(None), CUarray, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpy2DAsync_v2 = _libraries['libcuda.so'].cuMemcpy2DAsync_v2
    cuMemcpy2DAsync_v2.restype = CUresult
    cuMemcpy2DAsync_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st), CUstream]
except AttributeError:
    pass
try:
    cuMemcpy3DAsync_v2 = _libraries['libcuda.so'].cuMemcpy3DAsync_v2
    cuMemcpy3DAsync_v2.restype = CUresult
    cuMemcpy3DAsync_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUstream]
except AttributeError:
    pass
try:
    cuMemcpy3DPeerAsync = _libraries['libcuda.so'].cuMemcpy3DPeerAsync
    cuMemcpy3DPeerAsync.restype = CUresult
    cuMemcpy3DPeerAsync.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_PEER_st), CUstream]
except AttributeError:
    pass
try:
    cuMemsetD8_v2 = _libraries['libcuda.so'].cuMemsetD8_v2
    cuMemsetD8_v2.restype = CUresult
    cuMemsetD8_v2.argtypes = [CUdeviceptr, ctypes.c_ubyte, size_t]
except AttributeError:
    pass
try:
    cuMemsetD16_v2 = _libraries['libcuda.so'].cuMemsetD16_v2
    cuMemsetD16_v2.restype = CUresult
    cuMemsetD16_v2.argtypes = [CUdeviceptr, ctypes.c_uint16, size_t]
except AttributeError:
    pass
try:
    cuMemsetD32_v2 = _libraries['libcuda.so'].cuMemsetD32_v2
    cuMemsetD32_v2.restype = CUresult
    cuMemsetD32_v2.argtypes = [CUdeviceptr, ctypes.c_uint32, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D8_v2 = _libraries['libcuda.so'].cuMemsetD2D8_v2
    cuMemsetD2D8_v2.restype = CUresult
    cuMemsetD2D8_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D16_v2 = _libraries['libcuda.so'].cuMemsetD2D16_v2
    cuMemsetD2D16_v2.restype = CUresult
    cuMemsetD2D16_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D32_v2 = _libraries['libcuda.so'].cuMemsetD2D32_v2
    cuMemsetD2D32_v2.restype = CUresult
    cuMemsetD2D32_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD8Async = _libraries['libcuda.so'].cuMemsetD8Async
    cuMemsetD8Async.restype = CUresult
    cuMemsetD8Async.argtypes = [CUdeviceptr, ctypes.c_ubyte, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD16Async = _libraries['libcuda.so'].cuMemsetD16Async
    cuMemsetD16Async.restype = CUresult
    cuMemsetD16Async.argtypes = [CUdeviceptr, ctypes.c_uint16, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD32Async = _libraries['libcuda.so'].cuMemsetD32Async
    cuMemsetD32Async.restype = CUresult
    cuMemsetD32Async.argtypes = [CUdeviceptr, ctypes.c_uint32, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D8Async = _libraries['libcuda.so'].cuMemsetD2D8Async
    cuMemsetD2D8Async.restype = CUresult
    cuMemsetD2D8Async.argtypes = [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D16Async = _libraries['libcuda.so'].cuMemsetD2D16Async
    cuMemsetD2D16Async.restype = CUresult
    cuMemsetD2D16Async.argtypes = [CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D32Async = _libraries['libcuda.so'].cuMemsetD2D32Async
    cuMemsetD2D32Async.restype = CUresult
    cuMemsetD2D32Async.argtypes = [CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuArrayCreate_v2 = _libraries['libcuda.so'].cuArrayCreate_v2
    cuArrayCreate_v2.restype = CUresult
    cuArrayCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st)]
except AttributeError:
    pass
try:
    cuArrayGetDescriptor_v2 = _libraries['libcuda.so'].cuArrayGetDescriptor_v2
    cuArrayGetDescriptor_v2.restype = CUresult
    cuArrayGetDescriptor_v2.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st), CUarray]
except AttributeError:
    pass
try:
    cuArrayGetSparseProperties = _libraries['libcuda.so'].cuArrayGetSparseProperties
    cuArrayGetSparseProperties.restype = CUresult
    cuArrayGetSparseProperties.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st), CUarray]
except AttributeError:
    pass
try:
    cuMipmappedArrayGetSparseProperties = _libraries['libcuda.so'].cuMipmappedArrayGetSparseProperties
    cuMipmappedArrayGetSparseProperties.restype = CUresult
    cuMipmappedArrayGetSparseProperties.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st), CUmipmappedArray]
except AttributeError:
    pass
try:
    cuArrayGetPlane = _libraries['libcuda.so'].cuArrayGetPlane
    cuArrayGetPlane.restype = CUresult
    cuArrayGetPlane.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuArrayDestroy = _libraries['libcuda.so'].cuArrayDestroy
    cuArrayDestroy.restype = CUresult
    cuArrayDestroy.argtypes = [CUarray]
except AttributeError:
    pass
try:
    cuArray3DCreate_v2 = _libraries['libcuda.so'].cuArray3DCreate_v2
    cuArray3DCreate_v2.restype = CUresult
    cuArray3DCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st)]
except AttributeError:
    pass
try:
    cuArray3DGetDescriptor_v2 = _libraries['libcuda.so'].cuArray3DGetDescriptor_v2
    cuArray3DGetDescriptor_v2.restype = CUresult
    cuArray3DGetDescriptor_v2.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st), CUarray]
except AttributeError:
    pass
try:
    cuMipmappedArrayCreate = _libraries['libcuda.so'].cuMipmappedArrayCreate
    cuMipmappedArrayCreate.restype = CUresult
    cuMipmappedArrayCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMipmappedArrayGetLevel = _libraries['libcuda.so'].cuMipmappedArrayGetLevel
    cuMipmappedArrayGetLevel.restype = CUresult
    cuMipmappedArrayGetLevel.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUmipmappedArray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMipmappedArrayDestroy = _libraries['libcuda.so'].cuMipmappedArrayDestroy
    cuMipmappedArrayDestroy.restype = CUresult
    cuMipmappedArrayDestroy.argtypes = [CUmipmappedArray]
except AttributeError:
    pass
try:
    cuMemAddressReserve = _libraries['libcuda.so'].cuMemAddressReserve
    cuMemAddressReserve.restype = CUresult
    cuMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, size_t, CUdeviceptr, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemAddressFree = _libraries['libcuda.so'].cuMemAddressFree
    cuMemAddressFree.restype = CUresult
    cuMemAddressFree.argtypes = [CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemCreate = _libraries['libcuda.so'].cuMemCreate
    cuMemCreate.restype = CUresult
    cuMemCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, ctypes.POINTER(struct_CUmemAllocationProp_st), ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemRelease = _libraries['libcuda.so'].cuMemRelease
    cuMemRelease.restype = CUresult
    cuMemRelease.argtypes = [CUmemGenericAllocationHandle]
except AttributeError:
    pass
try:
    cuMemMap = _libraries['libcuda.so'].cuMemMap
    cuMemMap.restype = CUresult
    cuMemMap.argtypes = [CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemMapArrayAsync = _libraries['libcuda.so'].cuMemMapArrayAsync
    cuMemMapArrayAsync.restype = CUresult
    cuMemMapArrayAsync.argtypes = [ctypes.POINTER(struct_CUarrayMapInfo_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuMemUnmap = _libraries['libcuda.so'].cuMemUnmap
    cuMemUnmap.restype = CUresult
    cuMemUnmap.argtypes = [CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemSetAccess = _libraries['libcuda.so'].cuMemSetAccess
    cuMemSetAccess.restype = CUresult
    cuMemSetAccess.argtypes = [CUdeviceptr, size_t, ctypes.POINTER(struct_CUmemAccessDesc_st), size_t]
except AttributeError:
    pass
try:
    cuMemGetAccess = _libraries['libcuda.so'].cuMemGetAccess
    cuMemGetAccess.restype = CUresult
    cuMemGetAccess.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUmemLocation_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemExportToShareableHandle = _libraries['libcuda.so'].cuMemExportToShareableHandle
    cuMemExportToShareableHandle.restype = CUresult
    cuMemExportToShareableHandle.argtypes = [ctypes.POINTER(None), CUmemGenericAllocationHandle, CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemImportFromShareableHandle = _libraries['libcuda.so'].cuMemImportFromShareableHandle
    cuMemImportFromShareableHandle.restype = CUresult
    cuMemImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None), CUmemAllocationHandleType]
except AttributeError:
    pass
try:
    cuMemGetAllocationGranularity = _libraries['libcuda.so'].cuMemGetAllocationGranularity
    cuMemGetAllocationGranularity.restype = CUresult
    cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUmemAllocationProp_st), CUmemAllocationGranularity_flags]
except AttributeError:
    pass
try:
    cuMemGetAllocationPropertiesFromHandle = _libraries['libcuda.so'].cuMemGetAllocationPropertiesFromHandle
    cuMemGetAllocationPropertiesFromHandle.restype = CUresult
    cuMemGetAllocationPropertiesFromHandle.argtypes = [ctypes.POINTER(struct_CUmemAllocationProp_st), CUmemGenericAllocationHandle]
except AttributeError:
    pass
try:
    cuMemRetainAllocationHandle = _libraries['libcuda.so'].cuMemRetainAllocationHandle
    cuMemRetainAllocationHandle.restype = CUresult
    cuMemRetainAllocationHandle.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemFreeAsync = _libraries['libcuda.so'].cuMemFreeAsync
    cuMemFreeAsync.restype = CUresult
    cuMemFreeAsync.argtypes = [CUdeviceptr, CUstream]
except AttributeError:
    pass
try:
    cuMemAllocAsync = _libraries['libcuda.so'].cuMemAllocAsync
    cuMemAllocAsync.restype = CUresult
    cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemPoolTrimTo = _libraries['libcuda.so'].cuMemPoolTrimTo
    cuMemPoolTrimTo.restype = CUresult
    cuMemPoolTrimTo.argtypes = [CUmemoryPool, size_t]
except AttributeError:
    pass
try:
    cuMemPoolSetAttribute = _libraries['libcuda.so'].cuMemPoolSetAttribute
    cuMemPoolSetAttribute.restype = CUresult
    cuMemPoolSetAttribute.argtypes = [CUmemoryPool, CUmemPool_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemPoolGetAttribute = _libraries['libcuda.so'].cuMemPoolGetAttribute
    cuMemPoolGetAttribute.restype = CUresult
    cuMemPoolGetAttribute.argtypes = [CUmemoryPool, CUmemPool_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemPoolSetAccess = _libraries['libcuda.so'].cuMemPoolSetAccess
    cuMemPoolSetAccess.restype = CUresult
    cuMemPoolSetAccess.argtypes = [CUmemoryPool, ctypes.POINTER(struct_CUmemAccessDesc_st), size_t]
except AttributeError:
    pass
try:
    cuMemPoolGetAccess = _libraries['libcuda.so'].cuMemPoolGetAccess
    cuMemPoolGetAccess.restype = CUresult
    cuMemPoolGetAccess.argtypes = [ctypes.POINTER(CUmemAccess_flags_enum), CUmemoryPool, ctypes.POINTER(struct_CUmemLocation_st)]
except AttributeError:
    pass
try:
    cuMemPoolCreate = _libraries['libcuda.so'].cuMemPoolCreate
    cuMemPoolCreate.restype = CUresult
    cuMemPoolCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), ctypes.POINTER(struct_CUmemPoolProps_st)]
except AttributeError:
    pass
try:
    cuMemPoolDestroy = _libraries['libcuda.so'].cuMemPoolDestroy
    cuMemPoolDestroy.restype = CUresult
    cuMemPoolDestroy.argtypes = [CUmemoryPool]
except AttributeError:
    pass
try:
    cuMemAllocFromPoolAsync = _libraries['libcuda.so'].cuMemAllocFromPoolAsync
    cuMemAllocFromPoolAsync.restype = CUresult
    cuMemAllocFromPoolAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, CUmemoryPool, CUstream]
except AttributeError:
    pass
try:
    cuMemPoolExportToShareableHandle = _libraries['libcuda.so'].cuMemPoolExportToShareableHandle
    cuMemPoolExportToShareableHandle.restype = CUresult
    cuMemPoolExportToShareableHandle.argtypes = [ctypes.POINTER(None), CUmemoryPool, CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemPoolImportFromShareableHandle = _libraries['libcuda.so'].cuMemPoolImportFromShareableHandle
    cuMemPoolImportFromShareableHandle.restype = CUresult
    cuMemPoolImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), ctypes.POINTER(None), CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemPoolExportPointer = _libraries['libcuda.so'].cuMemPoolExportPointer
    cuMemPoolExportPointer.restype = CUresult
    cuMemPoolExportPointer.argtypes = [ctypes.POINTER(struct_CUmemPoolPtrExportData_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemPoolImportPointer = _libraries['libcuda.so'].cuMemPoolImportPointer
    cuMemPoolImportPointer.restype = CUresult
    cuMemPoolImportPointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUmemoryPool, ctypes.POINTER(struct_CUmemPoolPtrExportData_st)]
except AttributeError:
    pass
try:
    cuPointerGetAttribute = _libraries['libcuda.so'].cuPointerGetAttribute
    cuPointerGetAttribute.restype = CUresult
    cuPointerGetAttribute.argtypes = [ctypes.POINTER(None), CUpointer_attribute, CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemPrefetchAsync = _libraries['libcuda.so'].cuMemPrefetchAsync
    cuMemPrefetchAsync.restype = CUresult
    cuMemPrefetchAsync.argtypes = [CUdeviceptr, size_t, CUdevice, CUstream]
except AttributeError:
    pass
try:
    cuMemAdvise = _libraries['libcuda.so'].cuMemAdvise
    cuMemAdvise.restype = CUresult
    cuMemAdvise.argtypes = [CUdeviceptr, size_t, CUmem_advise, CUdevice]
except AttributeError:
    pass
try:
    cuMemRangeGetAttribute = _libraries['libcuda.so'].cuMemRangeGetAttribute
    cuMemRangeGetAttribute.restype = CUresult
    cuMemRangeGetAttribute.argtypes = [ctypes.POINTER(None), size_t, CUmem_range_attribute, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemRangeGetAttributes = _libraries['libcuda.so'].cuMemRangeGetAttributes
    cuMemRangeGetAttributes.restype = CUresult
    cuMemRangeGetAttributes.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(CUmem_range_attribute_enum), size_t, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuPointerSetAttribute = _libraries['libcuda.so'].cuPointerSetAttribute
    cuPointerSetAttribute.restype = CUresult
    cuPointerSetAttribute.argtypes = [ctypes.POINTER(None), CUpointer_attribute, CUdeviceptr]
except AttributeError:
    pass
try:
    cuPointerGetAttributes = _libraries['libcuda.so'].cuPointerGetAttributes
    cuPointerGetAttributes.restype = CUresult
    cuPointerGetAttributes.argtypes = [ctypes.c_uint32, ctypes.POINTER(CUpointer_attribute_enum), ctypes.POINTER(ctypes.POINTER(None)), CUdeviceptr]
except AttributeError:
    pass
try:
    cuStreamCreate = _libraries['libcuda.so'].cuStreamCreate
    cuStreamCreate.restype = CUresult
    cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUstream_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamCreateWithPriority = _libraries['libcuda.so'].cuStreamCreateWithPriority
    cuStreamCreateWithPriority.restype = CUresult
    cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUstream_st)), ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuStreamGetPriority = _libraries['libcuda.so'].cuStreamGetPriority
    cuStreamGetPriority.restype = CUresult
    cuStreamGetPriority.argtypes = [CUstream, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuStreamGetFlags = _libraries['libcuda.so'].cuStreamGetFlags
    cuStreamGetFlags.restype = CUresult
    cuStreamGetFlags.argtypes = [CUstream, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuStreamGetCtx = _libraries['libcuda.so'].cuStreamGetCtx
    cuStreamGetCtx.restype = CUresult
    cuStreamGetCtx.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuStreamWaitEvent = _libraries['libcuda.so'].cuStreamWaitEvent
    cuStreamWaitEvent.restype = CUresult
    cuStreamWaitEvent.argtypes = [CUstream, CUevent, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamAddCallback = _libraries['libcuda.so'].cuStreamAddCallback
    cuStreamAddCallback.restype = CUresult
    cuStreamAddCallback.argtypes = [CUstream, CUstreamCallback, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamBeginCapture_v2 = _libraries['libcuda.so'].cuStreamBeginCapture_v2
    cuStreamBeginCapture_v2.restype = CUresult
    cuStreamBeginCapture_v2.argtypes = [CUstream, CUstreamCaptureMode]
except AttributeError:
    pass
try:
    cuThreadExchangeStreamCaptureMode = _libraries['libcuda.so'].cuThreadExchangeStreamCaptureMode
    cuThreadExchangeStreamCaptureMode.restype = CUresult
    cuThreadExchangeStreamCaptureMode.argtypes = [ctypes.POINTER(CUstreamCaptureMode_enum)]
except AttributeError:
    pass
try:
    cuStreamEndCapture = _libraries['libcuda.so'].cuStreamEndCapture
    cuStreamEndCapture.restype = CUresult
    cuStreamEndCapture.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st))]
except AttributeError:
    pass
try:
    cuStreamIsCapturing = _libraries['libcuda.so'].cuStreamIsCapturing
    cuStreamIsCapturing.restype = CUresult
    cuStreamIsCapturing.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum)]
except AttributeError:
    pass
try:
    cuStreamGetCaptureInfo = _libraries['libcuda.so'].cuStreamGetCaptureInfo
    cuStreamGetCaptureInfo.restype = CUresult
    cuStreamGetCaptureInfo.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuStreamGetCaptureInfo_v2 = _libraries['libcuda.so'].cuStreamGetCaptureInfo_v2
    cuStreamGetCaptureInfo_v2.restype = CUresult
    cuStreamGetCaptureInfo_v2.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st))), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuStreamUpdateCaptureDependencies = _libraries['libcuda.so'].cuStreamUpdateCaptureDependencies
    cuStreamUpdateCaptureDependencies.restype = CUresult
    cuStreamUpdateCaptureDependencies.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamAttachMemAsync = _libraries['libcuda.so'].cuStreamAttachMemAsync
    cuStreamAttachMemAsync.restype = CUresult
    cuStreamAttachMemAsync.argtypes = [CUstream, CUdeviceptr, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamQuery = _libraries['libcuda.so'].cuStreamQuery
    cuStreamQuery.restype = CUresult
    cuStreamQuery.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamSynchronize = _libraries['libcuda.so'].cuStreamSynchronize
    cuStreamSynchronize.restype = CUresult
    cuStreamSynchronize.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamDestroy_v2 = _libraries['libcuda.so'].cuStreamDestroy_v2
    cuStreamDestroy_v2.restype = CUresult
    cuStreamDestroy_v2.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamCopyAttributes = _libraries['libcuda.so'].cuStreamCopyAttributes
    cuStreamCopyAttributes.restype = CUresult
    cuStreamCopyAttributes.argtypes = [CUstream, CUstream]
except AttributeError:
    pass
try:
    cuStreamGetAttribute = _libraries['libcuda.so'].cuStreamGetAttribute
    cuStreamGetAttribute.restype = CUresult
    cuStreamGetAttribute.argtypes = [CUstream, CUstreamAttrID, ctypes.POINTER(union_CUstreamAttrValue_union)]
except AttributeError:
    pass
try:
    cuStreamSetAttribute = _libraries['libcuda.so'].cuStreamSetAttribute
    cuStreamSetAttribute.restype = CUresult
    cuStreamSetAttribute.argtypes = [CUstream, CUstreamAttrID, ctypes.POINTER(union_CUstreamAttrValue_union)]
except AttributeError:
    pass
try:
    cuEventCreate = _libraries['libcuda.so'].cuEventCreate
    cuEventCreate.restype = CUresult
    cuEventCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUevent_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuEventRecord = _libraries['libcuda.so'].cuEventRecord
    cuEventRecord.restype = CUresult
    cuEventRecord.argtypes = [CUevent, CUstream]
except AttributeError:
    pass
try:
    cuEventRecordWithFlags = _libraries['libcuda.so'].cuEventRecordWithFlags
    cuEventRecordWithFlags.restype = CUresult
    cuEventRecordWithFlags.argtypes = [CUevent, CUstream, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuEventQuery = _libraries['libcuda.so'].cuEventQuery
    cuEventQuery.restype = CUresult
    cuEventQuery.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventSynchronize = _libraries['libcuda.so'].cuEventSynchronize
    cuEventSynchronize.restype = CUresult
    cuEventSynchronize.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventDestroy_v2 = _libraries['libcuda.so'].cuEventDestroy_v2
    cuEventDestroy_v2.restype = CUresult
    cuEventDestroy_v2.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventElapsedTime = _libraries['libcuda.so'].cuEventElapsedTime
    cuEventElapsedTime.restype = CUresult
    cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), CUevent, CUevent]
except AttributeError:
    pass
try:
    cuImportExternalMemory = _libraries['libcuda.so'].cuImportExternalMemory
    cuImportExternalMemory.restype = CUresult
    cuImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextMemory_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st)]
except AttributeError:
    pass
try:
    cuExternalMemoryGetMappedBuffer = _libraries['libcuda.so'].cuExternalMemoryGetMappedBuffer
    cuExternalMemoryGetMappedBuffer.restype = CUresult
    cuExternalMemoryGetMappedBuffer.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUexternalMemory, ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st)]
except AttributeError:
    pass
try:
    cuExternalMemoryGetMappedMipmappedArray = _libraries['libcuda.so'].cuExternalMemoryGetMappedMipmappedArray
    cuExternalMemoryGetMappedMipmappedArray.restype = CUresult
    cuExternalMemoryGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUexternalMemory, ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st)]
except AttributeError:
    pass
try:
    cuDestroyExternalMemory = _libraries['libcuda.so'].cuDestroyExternalMemory
    cuDestroyExternalMemory.restype = CUresult
    cuDestroyExternalMemory.argtypes = [CUexternalMemory]
except AttributeError:
    pass
try:
    cuImportExternalSemaphore = _libraries['libcuda.so'].cuImportExternalSemaphore
    cuImportExternalSemaphore.restype = CUresult
    cuImportExternalSemaphore.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st)]
except AttributeError:
    pass
try:
    cuSignalExternalSemaphoresAsync = _libraries['libcuda.so'].cuSignalExternalSemaphoresAsync
    cuSignalExternalSemaphoresAsync.restype = CUresult
    cuSignalExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuWaitExternalSemaphoresAsync = _libraries['libcuda.so'].cuWaitExternalSemaphoresAsync
    cuWaitExternalSemaphoresAsync.restype = CUresult
    cuWaitExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuDestroyExternalSemaphore = _libraries['libcuda.so'].cuDestroyExternalSemaphore
    cuDestroyExternalSemaphore.restype = CUresult
    cuDestroyExternalSemaphore.argtypes = [CUexternalSemaphore]
except AttributeError:
    pass
try:
    cuStreamWaitValue32 = _libraries['libcuda.so'].cuStreamWaitValue32
    cuStreamWaitValue32.restype = CUresult
    cuStreamWaitValue32.argtypes = [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWaitValue64 = _libraries['libcuda.so'].cuStreamWaitValue64
    cuStreamWaitValue64.restype = CUresult
    cuStreamWaitValue64.argtypes = [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWriteValue32 = _libraries['libcuda.so'].cuStreamWriteValue32
    cuStreamWriteValue32.restype = CUresult
    cuStreamWriteValue32.argtypes = [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWriteValue64 = _libraries['libcuda.so'].cuStreamWriteValue64
    cuStreamWriteValue64.restype = CUresult
    cuStreamWriteValue64.argtypes = [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamBatchMemOp = _libraries['libcuda.so'].cuStreamBatchMemOp
    cuStreamBatchMemOp.restype = CUresult
    cuStreamBatchMemOp.argtypes = [CUstream, ctypes.c_uint32, ctypes.POINTER(union_CUstreamBatchMemOpParams_union), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuFuncGetAttribute = _libraries['libcuda.so'].cuFuncGetAttribute
    cuFuncGetAttribute.restype = CUresult
    cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction_attribute, CUfunction]
except AttributeError:
    pass
try:
    cuFuncSetAttribute = _libraries['libcuda.so'].cuFuncSetAttribute
    cuFuncSetAttribute.restype = CUresult
    cuFuncSetAttribute.argtypes = [CUfunction, CUfunction_attribute, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuFuncSetCacheConfig = _libraries['libcuda.so'].cuFuncSetCacheConfig
    cuFuncSetCacheConfig.restype = CUresult
    cuFuncSetCacheConfig.argtypes = [CUfunction, CUfunc_cache]
except AttributeError:
    pass
try:
    cuFuncSetSharedMemConfig = _libraries['libcuda.so'].cuFuncSetSharedMemConfig
    cuFuncSetSharedMemConfig.restype = CUresult
    cuFuncSetSharedMemConfig.argtypes = [CUfunction, CUsharedconfig]
except AttributeError:
    pass
try:
    cuFuncGetModule = _libraries['libcuda.so'].cuFuncGetModule
    cuFuncGetModule.restype = CUresult
    cuFuncGetModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), CUfunction]
except AttributeError:
    pass
try:
    cuLaunchKernel = _libraries['libcuda.so'].cuLaunchKernel
    cuLaunchKernel.restype = CUresult
    cuLaunchKernel.argtypes = [CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLaunchCooperativeKernel = _libraries['libcuda.so'].cuLaunchCooperativeKernel
    cuLaunchCooperativeKernel.restype = CUresult
    cuLaunchCooperativeKernel.argtypes = [CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLaunchCooperativeKernelMultiDevice = _libraries['libcuda.so'].cuLaunchCooperativeKernelMultiDevice
    cuLaunchCooperativeKernelMultiDevice.restype = CUresult
    cuLaunchCooperativeKernelMultiDevice.argtypes = [ctypes.POINTER(struct_CUDA_LAUNCH_PARAMS_st), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuLaunchHostFunc = _libraries['libcuda.so'].cuLaunchHostFunc
    cuLaunchHostFunc.restype = CUresult
    cuLaunchHostFunc.argtypes = [CUstream, CUhostFn, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuFuncSetBlockShape = _libraries['libcuda.so'].cuFuncSetBlockShape
    cuFuncSetBlockShape.restype = CUresult
    cuFuncSetBlockShape.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuFuncSetSharedSize = _libraries['libcuda.so'].cuFuncSetSharedSize
    cuFuncSetSharedSize.restype = CUresult
    cuFuncSetSharedSize.argtypes = [CUfunction, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSetSize = _libraries['libcuda.so'].cuParamSetSize
    cuParamSetSize.restype = CUresult
    cuParamSetSize.argtypes = [CUfunction, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSeti = _libraries['libcuda.so'].cuParamSeti
    cuParamSeti.restype = CUresult
    cuParamSeti.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSetf = _libraries['libcuda.so'].cuParamSetf
    cuParamSetf.restype = CUresult
    cuParamSetf.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_float]
except AttributeError:
    pass
try:
    cuParamSetv = _libraries['libcuda.so'].cuParamSetv
    cuParamSetv.restype = CUresult
    cuParamSetv.argtypes = [CUfunction, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuLaunch = _libraries['libcuda.so'].cuLaunch
    cuLaunch.restype = CUresult
    cuLaunch.argtypes = [CUfunction]
except AttributeError:
    pass
try:
    cuLaunchGrid = _libraries['libcuda.so'].cuLaunchGrid
    cuLaunchGrid.restype = CUresult
    cuLaunchGrid.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuLaunchGridAsync = _libraries['libcuda.so'].cuLaunchGridAsync
    cuLaunchGridAsync.restype = CUresult
    cuLaunchGridAsync.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32, CUstream]
except AttributeError:
    pass
try:
    cuParamSetTexRef = _libraries['libcuda.so'].cuParamSetTexRef
    cuParamSetTexRef.restype = CUresult
    cuParamSetTexRef.argtypes = [CUfunction, ctypes.c_int32, CUtexref]
except AttributeError:
    pass
try:
    cuGraphCreate = _libraries['libcuda.so'].cuGraphCreate
    cuGraphCreate.restype = CUresult
    cuGraphCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphAddKernelNode = _libraries['libcuda.so'].cuGraphAddKernelNode
    cuGraphAddKernelNode.restype = CUresult
    cuGraphAddKernelNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeGetParams = _libraries['libcuda.so'].cuGraphKernelNodeGetParams
    cuGraphKernelNodeGetParams.restype = CUresult
    cuGraphKernelNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeSetParams = _libraries['libcuda.so'].cuGraphKernelNodeSetParams
    cuGraphKernelNodeSetParams.restype = CUresult
    cuGraphKernelNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemcpyNode = _libraries['libcuda.so'].cuGraphAddMemcpyNode
    cuGraphAddMemcpyNode.restype = CUresult
    cuGraphAddMemcpyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphMemcpyNodeGetParams = _libraries['libcuda.so'].cuGraphMemcpyNodeGetParams
    cuGraphMemcpyNodeGetParams.restype = CUresult
    cuGraphMemcpyNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuGraphMemcpyNodeSetParams = _libraries['libcuda.so'].cuGraphMemcpyNodeSetParams
    cuGraphMemcpyNodeSetParams.restype = CUresult
    cuGraphMemcpyNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemsetNode = _libraries['libcuda.so'].cuGraphAddMemsetNode
    cuGraphAddMemsetNode.restype = CUresult
    cuGraphAddMemsetNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphMemsetNodeGetParams = _libraries['libcuda.so'].cuGraphMemsetNodeGetParams
    cuGraphMemsetNodeGetParams.restype = CUresult
    cuGraphMemsetNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphMemsetNodeSetParams = _libraries['libcuda.so'].cuGraphMemsetNodeSetParams
    cuGraphMemsetNodeSetParams.restype = CUresult
    cuGraphMemsetNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddHostNode = _libraries['libcuda.so'].cuGraphAddHostNode
    cuGraphAddHostNode.restype = CUresult
    cuGraphAddHostNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphHostNodeGetParams = _libraries['libcuda.so'].cuGraphHostNodeGetParams
    cuGraphHostNodeGetParams.restype = CUresult
    cuGraphHostNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphHostNodeSetParams = _libraries['libcuda.so'].cuGraphHostNodeSetParams
    cuGraphHostNodeSetParams.restype = CUresult
    cuGraphHostNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddChildGraphNode = _libraries['libcuda.so'].cuGraphAddChildGraphNode
    cuGraphAddChildGraphNode.restype = CUresult
    cuGraphAddChildGraphNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUgraph]
except AttributeError:
    pass
try:
    cuGraphChildGraphNodeGetGraph = _libraries['libcuda.so'].cuGraphChildGraphNodeGetGraph
    cuGraphChildGraphNodeGetGraph.restype = CUresult
    cuGraphChildGraphNodeGetGraph.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st))]
except AttributeError:
    pass
try:
    cuGraphAddEmptyNode = _libraries['libcuda.so'].cuGraphAddEmptyNode
    cuGraphAddEmptyNode.restype = CUresult
    cuGraphAddEmptyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphAddEventRecordNode = _libraries['libcuda.so'].cuGraphAddEventRecordNode
    cuGraphAddEventRecordNode.restype = CUresult
    cuGraphAddEventRecordNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUevent]
except AttributeError:
    pass
try:
    cuGraphEventRecordNodeGetEvent = _libraries['libcuda.so'].cuGraphEventRecordNodeGetEvent
    cuGraphEventRecordNodeGetEvent.restype = CUresult
    cuGraphEventRecordNodeGetEvent.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUevent_st))]
except AttributeError:
    pass
try:
    cuGraphEventRecordNodeSetEvent = _libraries['libcuda.so'].cuGraphEventRecordNodeSetEvent
    cuGraphEventRecordNodeSetEvent.restype = CUresult
    cuGraphEventRecordNodeSetEvent.argtypes = [CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphAddEventWaitNode = _libraries['libcuda.so'].cuGraphAddEventWaitNode
    cuGraphAddEventWaitNode.restype = CUresult
    cuGraphAddEventWaitNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUevent]
except AttributeError:
    pass
try:
    cuGraphEventWaitNodeGetEvent = _libraries['libcuda.so'].cuGraphEventWaitNodeGetEvent
    cuGraphEventWaitNodeGetEvent.restype = CUresult
    cuGraphEventWaitNodeGetEvent.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUevent_st))]
except AttributeError:
    pass
try:
    cuGraphEventWaitNodeSetEvent = _libraries['libcuda.so'].cuGraphEventWaitNodeSetEvent
    cuGraphEventWaitNodeSetEvent.restype = CUresult
    cuGraphEventWaitNodeSetEvent.argtypes = [CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphAddExternalSemaphoresSignalNode = _libraries['libcuda.so'].cuGraphAddExternalSemaphoresSignalNode
    cuGraphAddExternalSemaphoresSignalNode.restype = CUresult
    cuGraphAddExternalSemaphoresSignalNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresSignalNodeGetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresSignalNodeGetParams
    cuGraphExternalSemaphoresSignalNodeGetParams.restype = CUresult
    cuGraphExternalSemaphoresSignalNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresSignalNodeSetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresSignalNodeSetParams
    cuGraphExternalSemaphoresSignalNodeSetParams.restype = CUresult
    cuGraphExternalSemaphoresSignalNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddExternalSemaphoresWaitNode = _libraries['libcuda.so'].cuGraphAddExternalSemaphoresWaitNode
    cuGraphAddExternalSemaphoresWaitNode.restype = CUresult
    cuGraphAddExternalSemaphoresWaitNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresWaitNodeGetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresWaitNodeGetParams
    cuGraphExternalSemaphoresWaitNodeGetParams.restype = CUresult
    cuGraphExternalSemaphoresWaitNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresWaitNodeSetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresWaitNodeSetParams
    cuGraphExternalSemaphoresWaitNodeSetParams.restype = CUresult
    cuGraphExternalSemaphoresWaitNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemAllocNode = _libraries['libcuda.so'].cuGraphAddMemAllocNode
    cuGraphAddMemAllocNode.restype = CUresult
    cuGraphAddMemAllocNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphMemAllocNodeGetParams = _libraries['libcuda.so'].cuGraphMemAllocNodeGetParams
    cuGraphMemAllocNodeGetParams.restype = CUresult
    cuGraphMemAllocNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemFreeNode = _libraries['libcuda.so'].cuGraphAddMemFreeNode
    cuGraphAddMemFreeNode.restype = CUresult
    cuGraphAddMemFreeNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUdeviceptr]
except AttributeError:
    pass
try:
    cuGraphMemFreeNodeGetParams = _libraries['libcuda.so'].cuGraphMemFreeNodeGetParams
    cuGraphMemFreeNodeGetParams.restype = CUresult
    cuGraphMemFreeNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuDeviceGraphMemTrim = _libraries['libcuda.so'].cuDeviceGraphMemTrim
    cuDeviceGraphMemTrim.restype = CUresult
    cuDeviceGraphMemTrim.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetGraphMemAttribute = _libraries['libcuda.so'].cuDeviceGetGraphMemAttribute
    cuDeviceGetGraphMemAttribute.restype = CUresult
    cuDeviceGetGraphMemAttribute.argtypes = [CUdevice, CUgraphMem_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuDeviceSetGraphMemAttribute = _libraries['libcuda.so'].cuDeviceSetGraphMemAttribute
    cuDeviceSetGraphMemAttribute.restype = CUresult
    cuDeviceSetGraphMemAttribute.argtypes = [CUdevice, CUgraphMem_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuGraphClone = _libraries['libcuda.so'].cuGraphClone
    cuGraphClone.restype = CUresult
    cuGraphClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), CUgraph]
except AttributeError:
    pass
try:
    cuGraphNodeFindInClone = _libraries['libcuda.so'].cuGraphNodeFindInClone
    cuGraphNodeFindInClone.restype = CUresult
    cuGraphNodeFindInClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraphNode, CUgraph]
except AttributeError:
    pass
try:
    cuGraphNodeGetType = _libraries['libcuda.so'].cuGraphNodeGetType
    cuGraphNodeGetType.restype = CUresult
    cuGraphNodeGetType.argtypes = [CUgraphNode, ctypes.POINTER(CUgraphNodeType_enum)]
except AttributeError:
    pass
try:
    cuGraphGetNodes = _libraries['libcuda.so'].cuGraphGetNodes
    cuGraphGetNodes.restype = CUresult
    cuGraphGetNodes.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphGetRootNodes = _libraries['libcuda.so'].cuGraphGetRootNodes
    cuGraphGetRootNodes.restype = CUresult
    cuGraphGetRootNodes.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphGetEdges = _libraries['libcuda.so'].cuGraphGetEdges
    cuGraphGetEdges.restype = CUresult
    cuGraphGetEdges.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphNodeGetDependencies = _libraries['libcuda.so'].cuGraphNodeGetDependencies
    cuGraphNodeGetDependencies.restype = CUresult
    cuGraphNodeGetDependencies.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphNodeGetDependentNodes = _libraries['libcuda.so'].cuGraphNodeGetDependentNodes
    cuGraphNodeGetDependentNodes.restype = CUresult
    cuGraphNodeGetDependentNodes.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphAddDependencies = _libraries['libcuda.so'].cuGraphAddDependencies
    cuGraphAddDependencies.restype = CUresult
    cuGraphAddDependencies.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphRemoveDependencies = _libraries['libcuda.so'].cuGraphRemoveDependencies
    cuGraphRemoveDependencies.restype = CUresult
    cuGraphRemoveDependencies.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphDestroyNode = _libraries['libcuda.so'].cuGraphDestroyNode
    cuGraphDestroyNode.restype = CUresult
    cuGraphDestroyNode.argtypes = [CUgraphNode]
except AttributeError:
    pass
try:
    cuGraphInstantiate_v2 = _libraries['libcuda.so'].cuGraphInstantiate_v2
    cuGraphInstantiate_v2.restype = CUresult
    cuGraphInstantiate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphExec_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    cuGraphInstantiateWithFlags = _libraries['libcuda.so'].cuGraphInstantiateWithFlags
    cuGraphInstantiateWithFlags.restype = CUresult
    cuGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphExec_st)), CUgraph, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuGraphExecKernelNodeSetParams = _libraries['libcuda.so'].cuGraphExecKernelNodeSetParams
    cuGraphExecKernelNodeSetParams.restype = CUresult
    cuGraphExecKernelNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecMemcpyNodeSetParams = _libraries['libcuda.so'].cuGraphExecMemcpyNodeSetParams
    cuGraphExecMemcpyNodeSetParams.restype = CUresult
    cuGraphExecMemcpyNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphExecMemsetNodeSetParams = _libraries['libcuda.so'].cuGraphExecMemsetNodeSetParams
    cuGraphExecMemsetNodeSetParams.restype = CUresult
    cuGraphExecMemsetNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphExecHostNodeSetParams = _libraries['libcuda.so'].cuGraphExecHostNodeSetParams
    cuGraphExecHostNodeSetParams.restype = CUresult
    cuGraphExecHostNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecChildGraphNodeSetParams = _libraries['libcuda.so'].cuGraphExecChildGraphNodeSetParams
    cuGraphExecChildGraphNodeSetParams.restype = CUresult
    cuGraphExecChildGraphNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, CUgraph]
except AttributeError:
    pass
try:
    cuGraphExecEventRecordNodeSetEvent = _libraries['libcuda.so'].cuGraphExecEventRecordNodeSetEvent
    cuGraphExecEventRecordNodeSetEvent.restype = CUresult
    cuGraphExecEventRecordNodeSetEvent.argtypes = [CUgraphExec, CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphExecEventWaitNodeSetEvent = _libraries['libcuda.so'].cuGraphExecEventWaitNodeSetEvent
    cuGraphExecEventWaitNodeSetEvent.restype = CUresult
    cuGraphExecEventWaitNodeSetEvent.argtypes = [CUgraphExec, CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphExecExternalSemaphoresSignalNodeSetParams = _libraries['libcuda.so'].cuGraphExecExternalSemaphoresSignalNodeSetParams
    cuGraphExecExternalSemaphoresSignalNodeSetParams.restype = CUresult
    cuGraphExecExternalSemaphoresSignalNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecExternalSemaphoresWaitNodeSetParams = _libraries['libcuda.so'].cuGraphExecExternalSemaphoresWaitNodeSetParams
    cuGraphExecExternalSemaphoresWaitNodeSetParams.restype = CUresult
    cuGraphExecExternalSemaphoresWaitNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphUpload = _libraries['libcuda.so'].cuGraphUpload
    cuGraphUpload.restype = CUresult
    cuGraphUpload.argtypes = [CUgraphExec, CUstream]
except AttributeError:
    pass
try:
    cuGraphLaunch = _libraries['libcuda.so'].cuGraphLaunch
    cuGraphLaunch.restype = CUresult
    cuGraphLaunch.argtypes = [CUgraphExec, CUstream]
except AttributeError:
    pass
try:
    cuGraphExecDestroy = _libraries['libcuda.so'].cuGraphExecDestroy
    cuGraphExecDestroy.restype = CUresult
    cuGraphExecDestroy.argtypes = [CUgraphExec]
except AttributeError:
    pass
try:
    cuGraphDestroy = _libraries['libcuda.so'].cuGraphDestroy
    cuGraphDestroy.restype = CUresult
    cuGraphDestroy.argtypes = [CUgraph]
except AttributeError:
    pass
try:
    cuGraphExecUpdate = _libraries['libcuda.so'].cuGraphExecUpdate
    cuGraphExecUpdate.restype = CUresult
    cuGraphExecUpdate.argtypes = [CUgraphExec, CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(CUgraphExecUpdateResult_enum)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeCopyAttributes = _libraries['libcuda.so'].cuGraphKernelNodeCopyAttributes
    cuGraphKernelNodeCopyAttributes.restype = CUresult
    cuGraphKernelNodeCopyAttributes.argtypes = [CUgraphNode, CUgraphNode]
except AttributeError:
    pass
try:
    cuGraphKernelNodeGetAttribute = _libraries['libcuda.so'].cuGraphKernelNodeGetAttribute
    cuGraphKernelNodeGetAttribute.restype = CUresult
    cuGraphKernelNodeGetAttribute.argtypes = [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(union_CUkernelNodeAttrValue_union)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeSetAttribute = _libraries['libcuda.so'].cuGraphKernelNodeSetAttribute
    cuGraphKernelNodeSetAttribute.restype = CUresult
    cuGraphKernelNodeSetAttribute.argtypes = [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(union_CUkernelNodeAttrValue_union)]
except AttributeError:
    pass
try:
    cuGraphDebugDotPrint = _libraries['libcuda.so'].cuGraphDebugDotPrint
    cuGraphDebugDotPrint.restype = CUresult
    cuGraphDebugDotPrint.argtypes = [CUgraph, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectCreate = _libraries['libcuda.so'].cuUserObjectCreate
    cuUserObjectCreate.restype = CUresult
    cuUserObjectCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUuserObject_st)), ctypes.POINTER(None), CUhostFn, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectRetain = _libraries['libcuda.so'].cuUserObjectRetain
    cuUserObjectRetain.restype = CUresult
    cuUserObjectRetain.argtypes = [CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectRelease = _libraries['libcuda.so'].cuUserObjectRelease
    cuUserObjectRelease.restype = CUresult
    cuUserObjectRelease.argtypes = [CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphRetainUserObject = _libraries['libcuda.so'].cuGraphRetainUserObject
    cuGraphRetainUserObject.restype = CUresult
    cuGraphRetainUserObject.argtypes = [CUgraph, CUuserObject, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphReleaseUserObject = _libraries['libcuda.so'].cuGraphReleaseUserObject
    cuGraphReleaseUserObject.restype = CUresult
    cuGraphReleaseUserObject.argtypes = [CUgraph, CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyMaxActiveBlocksPerMultiprocessor = _libraries['libcuda.so'].cuOccupancyMaxActiveBlocksPerMultiprocessor
    cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = CUresult
    cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = _libraries['libcuda.so'].cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.restype = CUresult
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyMaxPotentialBlockSize = _libraries['libcuda.so'].cuOccupancyMaxPotentialBlockSize
    cuOccupancyMaxPotentialBlockSize.restype = CUresult
    cuOccupancyMaxPotentialBlockSize.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuOccupancyMaxPotentialBlockSizeWithFlags = _libraries['libcuda.so'].cuOccupancyMaxPotentialBlockSizeWithFlags
    cuOccupancyMaxPotentialBlockSizeWithFlags.restype = CUresult
    cuOccupancyMaxPotentialBlockSizeWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyAvailableDynamicSMemPerBlock = _libraries['libcuda.so'].cuOccupancyAvailableDynamicSMemPerBlock
    cuOccupancyAvailableDynamicSMemPerBlock.restype = CUresult
    cuOccupancyAvailableDynamicSMemPerBlock.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUfunction, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefSetArray = _libraries['libcuda.so'].cuTexRefSetArray
    cuTexRefSetArray.restype = CUresult
    cuTexRefSetArray.argtypes = [CUtexref, CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetMipmappedArray = _libraries['libcuda.so'].cuTexRefSetMipmappedArray
    cuTexRefSetMipmappedArray.restype = CUresult
    cuTexRefSetMipmappedArray.argtypes = [CUtexref, CUmipmappedArray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetAddress_v2 = _libraries['libcuda.so'].cuTexRefSetAddress_v2
    cuTexRefSetAddress_v2.restype = CUresult
    cuTexRefSetAddress_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUtexref, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuTexRefSetAddress2D_v3 = _libraries['libcuda.so'].cuTexRefSetAddress2D_v3
    cuTexRefSetAddress2D_v3.restype = CUresult
    cuTexRefSetAddress2D_v3.argtypes = [CUtexref, ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st), CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuTexRefSetFormat = _libraries['libcuda.so'].cuTexRefSetFormat
    cuTexRefSetFormat.restype = CUresult
    cuTexRefSetFormat.argtypes = [CUtexref, CUarray_format, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefSetAddressMode = _libraries['libcuda.so'].cuTexRefSetAddressMode
    cuTexRefSetAddressMode.restype = CUresult
    cuTexRefSetAddressMode.argtypes = [CUtexref, ctypes.c_int32, CUaddress_mode]
except AttributeError:
    pass
try:
    cuTexRefSetFilterMode = _libraries['libcuda.so'].cuTexRefSetFilterMode
    cuTexRefSetFilterMode.restype = CUresult
    cuTexRefSetFilterMode.argtypes = [CUtexref, CUfilter_mode]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapFilterMode = _libraries['libcuda.so'].cuTexRefSetMipmapFilterMode
    cuTexRefSetMipmapFilterMode.restype = CUresult
    cuTexRefSetMipmapFilterMode.argtypes = [CUtexref, CUfilter_mode]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapLevelBias = _libraries['libcuda.so'].cuTexRefSetMipmapLevelBias
    cuTexRefSetMipmapLevelBias.restype = CUresult
    cuTexRefSetMipmapLevelBias.argtypes = [CUtexref, ctypes.c_float]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapLevelClamp = _libraries['libcuda.so'].cuTexRefSetMipmapLevelClamp
    cuTexRefSetMipmapLevelClamp.restype = CUresult
    cuTexRefSetMipmapLevelClamp.argtypes = [CUtexref, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    cuTexRefSetMaxAnisotropy = _libraries['libcuda.so'].cuTexRefSetMaxAnisotropy
    cuTexRefSetMaxAnisotropy.restype = CUresult
    cuTexRefSetMaxAnisotropy.argtypes = [CUtexref, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetBorderColor = _libraries['libcuda.so'].cuTexRefSetBorderColor
    cuTexRefSetBorderColor.restype = CUresult
    cuTexRefSetBorderColor.argtypes = [CUtexref, ctypes.POINTER(ctypes.c_float)]
except AttributeError:
    pass
try:
    cuTexRefSetFlags = _libraries['libcuda.so'].cuTexRefSetFlags
    cuTexRefSetFlags.restype = CUresult
    cuTexRefSetFlags.argtypes = [CUtexref, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefGetAddress_v2 = _libraries['libcuda.so'].cuTexRefGetAddress_v2
    cuTexRefGetAddress_v2.restype = CUresult
    cuTexRefGetAddress_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetArray = _libraries['libcuda.so'].cuTexRefGetArray
    cuTexRefGetArray.restype = CUresult
    cuTexRefGetArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmappedArray = _libraries['libcuda.so'].cuTexRefGetMipmappedArray
    cuTexRefGetMipmappedArray.restype = CUresult
    cuTexRefGetMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetAddressMode = _libraries['libcuda.so'].cuTexRefGetAddressMode
    cuTexRefGetAddressMode.restype = CUresult
    cuTexRefGetAddressMode.argtypes = [ctypes.POINTER(CUaddress_mode_enum), CUtexref, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefGetFilterMode = _libraries['libcuda.so'].cuTexRefGetFilterMode
    cuTexRefGetFilterMode.restype = CUresult
    cuTexRefGetFilterMode.argtypes = [ctypes.POINTER(CUfilter_mode_enum), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetFormat = _libraries['libcuda.so'].cuTexRefGetFormat
    cuTexRefGetFormat.restype = CUresult
    cuTexRefGetFormat.argtypes = [ctypes.POINTER(CUarray_format_enum), ctypes.POINTER(ctypes.c_int32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapFilterMode = _libraries['libcuda.so'].cuTexRefGetMipmapFilterMode
    cuTexRefGetMipmapFilterMode.restype = CUresult
    cuTexRefGetMipmapFilterMode.argtypes = [ctypes.POINTER(CUfilter_mode_enum), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapLevelBias = _libraries['libcuda.so'].cuTexRefGetMipmapLevelBias
    cuTexRefGetMipmapLevelBias.restype = CUresult
    cuTexRefGetMipmapLevelBias.argtypes = [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapLevelClamp = _libraries['libcuda.so'].cuTexRefGetMipmapLevelClamp
    cuTexRefGetMipmapLevelClamp.restype = CUresult
    cuTexRefGetMipmapLevelClamp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMaxAnisotropy = _libraries['libcuda.so'].cuTexRefGetMaxAnisotropy
    cuTexRefGetMaxAnisotropy.restype = CUresult
    cuTexRefGetMaxAnisotropy.argtypes = [ctypes.POINTER(ctypes.c_int32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetBorderColor = _libraries['libcuda.so'].cuTexRefGetBorderColor
    cuTexRefGetBorderColor.restype = CUresult
    cuTexRefGetBorderColor.argtypes = [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetFlags = _libraries['libcuda.so'].cuTexRefGetFlags
    cuTexRefGetFlags.restype = CUresult
    cuTexRefGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefCreate = _libraries['libcuda.so'].cuTexRefCreate
    cuTexRefCreate.restype = CUresult
    cuTexRefCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUtexref_st))]
except AttributeError:
    pass
try:
    cuTexRefDestroy = _libraries['libcuda.so'].cuTexRefDestroy
    cuTexRefDestroy.restype = CUresult
    cuTexRefDestroy.argtypes = [CUtexref]
except AttributeError:
    pass
try:
    cuSurfRefSetArray = _libraries['libcuda.so'].cuSurfRefSetArray
    cuSurfRefSetArray.restype = CUresult
    cuSurfRefSetArray.argtypes = [CUsurfref, CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuSurfRefGetArray = _libraries['libcuda.so'].cuSurfRefGetArray
    cuSurfRefGetArray.restype = CUresult
    cuSurfRefGetArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUsurfref]
except AttributeError:
    pass
try:
    cuTexObjectCreate = _libraries['libcuda.so'].cuTexObjectCreate
    cuTexObjectCreate.restype = CUresult
    cuTexObjectCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), ctypes.POINTER(struct_CUDA_TEXTURE_DESC_st), ctypes.POINTER(struct_CUDA_RESOURCE_VIEW_DESC_st)]
except AttributeError:
    pass
try:
    cuTexObjectDestroy = _libraries['libcuda.so'].cuTexObjectDestroy
    cuTexObjectDestroy.restype = CUresult
    cuTexObjectDestroy.argtypes = [CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetResourceDesc = _libraries['libcuda.so'].cuTexObjectGetResourceDesc
    cuTexObjectGetResourceDesc.restype = CUresult
    cuTexObjectGetResourceDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetTextureDesc = _libraries['libcuda.so'].cuTexObjectGetTextureDesc
    cuTexObjectGetTextureDesc.restype = CUresult
    cuTexObjectGetTextureDesc.argtypes = [ctypes.POINTER(struct_CUDA_TEXTURE_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetResourceViewDesc = _libraries['libcuda.so'].cuTexObjectGetResourceViewDesc
    cuTexObjectGetResourceViewDesc.restype = CUresult
    cuTexObjectGetResourceViewDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_VIEW_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuSurfObjectCreate = _libraries['libcuda.so'].cuSurfObjectCreate
    cuSurfObjectCreate.restype = CUresult
    cuSurfObjectCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st)]
except AttributeError:
    pass
try:
    cuSurfObjectDestroy = _libraries['libcuda.so'].cuSurfObjectDestroy
    cuSurfObjectDestroy.restype = CUresult
    cuSurfObjectDestroy.argtypes = [CUsurfObject]
except AttributeError:
    pass
try:
    cuSurfObjectGetResourceDesc = _libraries['libcuda.so'].cuSurfObjectGetResourceDesc
    cuSurfObjectGetResourceDesc.restype = CUresult
    cuSurfObjectGetResourceDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), CUsurfObject]
except AttributeError:
    pass
try:
    cuDeviceCanAccessPeer = _libraries['libcuda.so'].cuDeviceCanAccessPeer
    cuDeviceCanAccessPeer.restype = CUresult
    cuDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice, CUdevice]
except AttributeError:
    pass
try:
    cuCtxEnablePeerAccess = _libraries['libcuda.so'].cuCtxEnablePeerAccess
    cuCtxEnablePeerAccess.restype = CUresult
    cuCtxEnablePeerAccess.argtypes = [CUcontext, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuCtxDisablePeerAccess = _libraries['libcuda.so'].cuCtxDisablePeerAccess
    cuCtxDisablePeerAccess.restype = CUresult
    cuCtxDisablePeerAccess.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuDeviceGetP2PAttribute = _libraries['libcuda.so'].cuDeviceGetP2PAttribute
    cuDeviceGetP2PAttribute.restype = CUresult
    cuDeviceGetP2PAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice_P2PAttribute, CUdevice, CUdevice]
except AttributeError:
    pass
try:
    cuGraphicsUnregisterResource = _libraries['libcuda.so'].cuGraphicsUnregisterResource
    cuGraphicsUnregisterResource.restype = CUresult
    cuGraphicsUnregisterResource.argtypes = [CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsSubResourceGetMappedArray = _libraries['libcuda.so'].cuGraphicsSubResourceGetMappedArray
    cuGraphicsSubResourceGetMappedArray.restype = CUresult
    cuGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUgraphicsResource, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphicsResourceGetMappedMipmappedArray = _libraries['libcuda.so'].cuGraphicsResourceGetMappedMipmappedArray
    cuGraphicsResourceGetMappedMipmappedArray.restype = CUresult
    cuGraphicsResourceGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsResourceGetMappedPointer_v2 = _libraries['libcuda.so'].cuGraphicsResourceGetMappedPointer_v2
    cuGraphicsResourceGetMappedPointer_v2.restype = CUresult
    cuGraphicsResourceGetMappedPointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsResourceSetMapFlags_v2 = _libraries['libcuda.so'].cuGraphicsResourceSetMapFlags_v2
    cuGraphicsResourceSetMapFlags_v2.restype = CUresult
    cuGraphicsResourceSetMapFlags_v2.argtypes = [CUgraphicsResource, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphicsMapResources = _libraries['libcuda.so'].cuGraphicsMapResources
    cuGraphicsMapResources.restype = CUresult
    cuGraphicsMapResources.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_CUgraphicsResource_st)), CUstream]
except AttributeError:
    pass
try:
    cuGraphicsUnmapResources = _libraries['libcuda.so'].cuGraphicsUnmapResources
    cuGraphicsUnmapResources.restype = CUresult
    cuGraphicsUnmapResources.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_CUgraphicsResource_st)), CUstream]
except AttributeError:
    pass
try:
    cuGetProcAddress = _libraries['libcuda.so'].cuGetProcAddress
    cuGetProcAddress.restype = CUresult
    cuGetProcAddress.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None)), ctypes.c_int32, cuuint64_t]
except AttributeError:
    pass
try:
    cuGetExportTable = _libraries['libcuda.so'].cuGetExportTable
    cuGetExportTable.restype = CUresult
    cuGetExportTable.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_CUuuid_st)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nvrtcResult'
c__EA_nvrtcResult__enumvalues = {
    0: 'NVRTC_SUCCESS',
    1: 'NVRTC_ERROR_OUT_OF_MEMORY',
    2: 'NVRTC_ERROR_PROGRAM_CREATION_FAILURE',
    3: 'NVRTC_ERROR_INVALID_INPUT',
    4: 'NVRTC_ERROR_INVALID_PROGRAM',
    5: 'NVRTC_ERROR_INVALID_OPTION',
    6: 'NVRTC_ERROR_COMPILATION',
    7: 'NVRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    8: 'NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    9: 'NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    10: 'NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    11: 'NVRTC_ERROR_INTERNAL_ERROR',
}
NVRTC_SUCCESS = 0
NVRTC_ERROR_OUT_OF_MEMORY = 1
NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
NVRTC_ERROR_INVALID_INPUT = 3
NVRTC_ERROR_INVALID_PROGRAM = 4
NVRTC_ERROR_INVALID_OPTION = 5
NVRTC_ERROR_COMPILATION = 6
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
NVRTC_ERROR_INTERNAL_ERROR = 11
c__EA_nvrtcResult = ctypes.c_uint32 # enum
nvrtcResult = c__EA_nvrtcResult
nvrtcResult__enumvalues = c__EA_nvrtcResult__enumvalues
try:
    nvrtcGetErrorString = _libraries['libnvrtc.so'].nvrtcGetErrorString
    nvrtcGetErrorString.restype = ctypes.POINTER(ctypes.c_char)
    nvrtcGetErrorString.argtypes = [nvrtcResult]
except AttributeError:
    pass
try:
    nvrtcVersion = _libraries['libnvrtc.so'].nvrtcVersion
    nvrtcVersion.restype = nvrtcResult
    nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    nvrtcGetNumSupportedArchs = _libraries['libnvrtc.so'].nvrtcGetNumSupportedArchs
    nvrtcGetNumSupportedArchs.restype = nvrtcResult
    nvrtcGetNumSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    nvrtcGetSupportedArchs = _libraries['libnvrtc.so'].nvrtcGetSupportedArchs
    nvrtcGetSupportedArchs.restype = nvrtcResult
    nvrtcGetSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
class struct__nvrtcProgram(Structure):
    pass

nvrtcProgram = ctypes.POINTER(struct__nvrtcProgram)
try:
    nvrtcCreateProgram = _libraries['libnvrtc.so'].nvrtcCreateProgram
    nvrtcCreateProgram.restype = nvrtcResult
    nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    nvrtcDestroyProgram = _libraries['libnvrtc.so'].nvrtcDestroyProgram
    nvrtcDestroyProgram.restype = nvrtcResult
    nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__nvrtcProgram))]
except AttributeError:
    pass
try:
    nvrtcCompileProgram = _libraries['libnvrtc.so'].nvrtcCompileProgram
    nvrtcCompileProgram.restype = nvrtcResult
    nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    nvrtcGetPTXSize = _libraries['libnvrtc.so'].nvrtcGetPTXSize
    nvrtcGetPTXSize.restype = nvrtcResult
    nvrtcGetPTXSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetPTX = _libraries['libnvrtc.so'].nvrtcGetPTX
    nvrtcGetPTX.restype = nvrtcResult
    nvrtcGetPTX.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetCUBINSize = _libraries['libnvrtc.so'].nvrtcGetCUBINSize
    nvrtcGetCUBINSize.restype = nvrtcResult
    nvrtcGetCUBINSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetCUBIN = _libraries['libnvrtc.so'].nvrtcGetCUBIN
    nvrtcGetCUBIN.restype = nvrtcResult
    nvrtcGetCUBIN.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetNVVMSize = _libraries['libnvrtc.so'].nvrtcGetNVVMSize
    nvrtcGetNVVMSize.restype = nvrtcResult
    nvrtcGetNVVMSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetNVVM = _libraries['libnvrtc.so'].nvrtcGetNVVM
    nvrtcGetNVVM.restype = nvrtcResult
    nvrtcGetNVVM.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetProgramLogSize = _libraries['libnvrtc.so'].nvrtcGetProgramLogSize
    nvrtcGetProgramLogSize.restype = nvrtcResult
    nvrtcGetProgramLogSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nvrtcGetProgramLog = _libraries['libnvrtc.so'].nvrtcGetProgramLog
    nvrtcGetProgramLog.restype = nvrtcResult
    nvrtcGetProgramLog.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcAddNameExpression = _libraries['libnvrtc.so'].nvrtcAddNameExpression
    nvrtcAddNameExpression.restype = nvrtcResult
    nvrtcAddNameExpression.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nvrtcGetLoweredName = _libraries['libnvrtc.so'].nvrtcGetLoweredName
    nvrtcGetLoweredName.restype = nvrtcResult
    nvrtcGetLoweredName.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
__all__ = \
    ['CUDA_ARRAY3D_DESCRIPTOR', 'CUDA_ARRAY3D_DESCRIPTOR_v2',
    'CUDA_ARRAY_DESCRIPTOR', 'CUDA_ARRAY_DESCRIPTOR_v2',
    'CUDA_ARRAY_SPARSE_PROPERTIES', 'CUDA_ARRAY_SPARSE_PROPERTIES_v1',
    'CUDA_ERROR_ALREADY_ACQUIRED', 'CUDA_ERROR_ALREADY_MAPPED',
    'CUDA_ERROR_ARRAY_IS_MAPPED', 'CUDA_ERROR_ASSERT',
    'CUDA_ERROR_CAPTURED_EVENT',
    'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    'CUDA_ERROR_DEINITIALIZED', 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    'CUDA_ERROR_ECC_UNCORRECTABLE', 'CUDA_ERROR_EXTERNAL_DEVICE',
    'CUDA_ERROR_FILE_NOT_FOUND',
    'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    'CUDA_ERROR_HARDWARE_STACK_ERROR',
    'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    'CUDA_ERROR_ILLEGAL_ADDRESS', 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    'CUDA_ERROR_ILLEGAL_STATE', 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    'CUDA_ERROR_INVALID_CONTEXT', 'CUDA_ERROR_INVALID_DEVICE',
    'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    'CUDA_ERROR_INVALID_HANDLE', 'CUDA_ERROR_INVALID_IMAGE',
    'CUDA_ERROR_INVALID_PC', 'CUDA_ERROR_INVALID_PTX',
    'CUDA_ERROR_INVALID_SOURCE', 'CUDA_ERROR_INVALID_VALUE',
    'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    'CUDA_ERROR_JIT_COMPILER_NOT_FOUND', 'CUDA_ERROR_LAUNCH_FAILED',
    'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES', 'CUDA_ERROR_LAUNCH_TIMEOUT',
    'CUDA_ERROR_MAP_FAILED', 'CUDA_ERROR_MISALIGNED_ADDRESS',
    'CUDA_ERROR_MPS_CONNECTION_FAILED',
    'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    'CUDA_ERROR_MPS_RPC_FAILURE', 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    'CUDA_ERROR_NOT_FOUND', 'CUDA_ERROR_NOT_INITIALIZED',
    'CUDA_ERROR_NOT_MAPPED', 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    'CUDA_ERROR_NOT_MAPPED_AS_POINTER', 'CUDA_ERROR_NOT_PERMITTED',
    'CUDA_ERROR_NOT_READY', 'CUDA_ERROR_NOT_SUPPORTED',
    'CUDA_ERROR_NO_BINARY_FOR_GPU', 'CUDA_ERROR_NO_DEVICE',
    'CUDA_ERROR_NVLINK_UNCORRECTABLE', 'CUDA_ERROR_OPERATING_SYSTEM',
    'CUDA_ERROR_OUT_OF_MEMORY',
    'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    'CUDA_ERROR_PROFILER_DISABLED',
    'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    'CUDA_ERROR_STUB_LIBRARY', 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    'CUDA_ERROR_SYSTEM_NOT_READY', 'CUDA_ERROR_TIMEOUT',
    'CUDA_ERROR_TOO_MANY_PEERS', 'CUDA_ERROR_UNKNOWN',
    'CUDA_ERROR_UNMAP_FAILED', 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    'CUDA_ERROR_UNSUPPORTED_LIMIT',
    'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
    'CUDA_EXTERNAL_MEMORY_BUFFER_DESC',
    'CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1',
    'CUDA_EXTERNAL_MEMORY_HANDLE_DESC',
    'CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1',
    'CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC',
    'CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1',
    'CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC',
    'CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1',
    'CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS',
    'CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1',
    'CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS',
    'CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1',
    'CUDA_EXT_SEM_SIGNAL_NODE_PARAMS',
    'CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1',
    'CUDA_EXT_SEM_WAIT_NODE_PARAMS',
    'CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1',
    'CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH',
    'CUDA_HOST_NODE_PARAMS', 'CUDA_HOST_NODE_PARAMS_v1',
    'CUDA_KERNEL_NODE_PARAMS', 'CUDA_KERNEL_NODE_PARAMS_v1',
    'CUDA_LAUNCH_PARAMS', 'CUDA_LAUNCH_PARAMS_v1', 'CUDA_MEMCPY2D',
    'CUDA_MEMCPY2D_v2', 'CUDA_MEMCPY3D', 'CUDA_MEMCPY3D_PEER',
    'CUDA_MEMCPY3D_PEER_v1', 'CUDA_MEMCPY3D_v2',
    'CUDA_MEMSET_NODE_PARAMS', 'CUDA_MEMSET_NODE_PARAMS_v1',
    'CUDA_MEM_ALLOC_NODE_PARAMS',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS__enumvalues',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum',
    'CUDA_POINTER_ATTRIBUTE_P2P_TOKENS',
    'CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1', 'CUDA_RESOURCE_DESC',
    'CUDA_RESOURCE_DESC_v1', 'CUDA_RESOURCE_VIEW_DESC',
    'CUDA_RESOURCE_VIEW_DESC_v1', 'CUDA_SUCCESS', 'CUDA_TEXTURE_DESC',
    'CUDA_TEXTURE_DESC_v1', 'CUGPUDirectRDMAWritesOrdering',
    'CUGPUDirectRDMAWritesOrdering__enumvalues',
    'CUGPUDirectRDMAWritesOrdering_enum', 'CU_ACCESS_PROPERTY_NORMAL',
    'CU_ACCESS_PROPERTY_PERSISTING', 'CU_ACCESS_PROPERTY_STREAMING',
    'CU_AD_FORMAT_BC1_UNORM', 'CU_AD_FORMAT_BC1_UNORM_SRGB',
    'CU_AD_FORMAT_BC2_UNORM', 'CU_AD_FORMAT_BC2_UNORM_SRGB',
    'CU_AD_FORMAT_BC3_UNORM', 'CU_AD_FORMAT_BC3_UNORM_SRGB',
    'CU_AD_FORMAT_BC4_SNORM', 'CU_AD_FORMAT_BC4_UNORM',
    'CU_AD_FORMAT_BC5_SNORM', 'CU_AD_FORMAT_BC5_UNORM',
    'CU_AD_FORMAT_BC6H_SF16', 'CU_AD_FORMAT_BC6H_UF16',
    'CU_AD_FORMAT_BC7_UNORM', 'CU_AD_FORMAT_BC7_UNORM_SRGB',
    'CU_AD_FORMAT_FLOAT', 'CU_AD_FORMAT_HALF', 'CU_AD_FORMAT_NV12',
    'CU_AD_FORMAT_SIGNED_INT16', 'CU_AD_FORMAT_SIGNED_INT32',
    'CU_AD_FORMAT_SIGNED_INT8', 'CU_AD_FORMAT_SNORM_INT16X1',
    'CU_AD_FORMAT_SNORM_INT16X2', 'CU_AD_FORMAT_SNORM_INT16X4',
    'CU_AD_FORMAT_SNORM_INT8X1', 'CU_AD_FORMAT_SNORM_INT8X2',
    'CU_AD_FORMAT_SNORM_INT8X4', 'CU_AD_FORMAT_UNORM_INT16X1',
    'CU_AD_FORMAT_UNORM_INT16X2', 'CU_AD_FORMAT_UNORM_INT16X4',
    'CU_AD_FORMAT_UNORM_INT8X1', 'CU_AD_FORMAT_UNORM_INT8X2',
    'CU_AD_FORMAT_UNORM_INT8X4', 'CU_AD_FORMAT_UNSIGNED_INT16',
    'CU_AD_FORMAT_UNSIGNED_INT32', 'CU_AD_FORMAT_UNSIGNED_INT8',
    'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL',
    'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL',
    'CU_COMPUTEMODE_DEFAULT', 'CU_COMPUTEMODE_EXCLUSIVE_PROCESS',
    'CU_COMPUTEMODE_PROHIBITED', 'CU_CTX_BLOCKING_SYNC',
    'CU_CTX_FLAGS_MASK', 'CU_CTX_LMEM_RESIZE_TO_MAX',
    'CU_CTX_MAP_HOST', 'CU_CTX_SCHED_AUTO',
    'CU_CTX_SCHED_BLOCKING_SYNC', 'CU_CTX_SCHED_MASK',
    'CU_CTX_SCHED_SPIN', 'CU_CTX_SCHED_YIELD',
    'CU_CUBEMAP_FACE_NEGATIVE_X', 'CU_CUBEMAP_FACE_NEGATIVE_Y',
    'CU_CUBEMAP_FACE_NEGATIVE_Z', 'CU_CUBEMAP_FACE_POSITIVE_X',
    'CU_CUBEMAP_FACE_POSITIVE_Y', 'CU_CUBEMAP_FACE_POSITIVE_Z',
    'CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT',
    'CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES',
    'CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY',
    'CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR',
    'CU_DEVICE_ATTRIBUTE_CLOCK_RATE',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_MODE',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS',
    'CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS',
    'CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH',
    'CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH',
    'CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST',
    'CU_DEVICE_ATTRIBUTE_ECC_ENABLED',
    'CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING',
    'CU_DEVICE_ATTRIBUTE_GPU_OVERLAP',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_INTEGRATED',
    'CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT',
    'CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE',
    'CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY', 'CU_DEVICE_ATTRIBUTE_MAX',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z',
    'CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE',
    'CU_DEVICE_ATTRIBUTE_MAX_PITCH',
    'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE',
    'CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES',
    'CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT',
    'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD',
    'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID',
    'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS',
    'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES',
    'CU_DEVICE_ATTRIBUTE_PCI_BUS_ID',
    'CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID',
    'CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID',
    'CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO',
    'CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TCC_DRIVER',
    'CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY',
    'CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING',
    'CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK',
    'CU_EVENT_BLOCKING_SYNC', 'CU_EVENT_DEFAULT',
    'CU_EVENT_DISABLE_TIMING', 'CU_EVENT_INTERPROCESS',
    'CU_EVENT_RECORD_DEFAULT', 'CU_EVENT_RECORD_EXTERNAL',
    'CU_EVENT_WAIT_DEFAULT', 'CU_EVENT_WAIT_EXTERNAL',
    'CU_EXEC_AFFINITY_TYPE_MAX', 'CU_EXEC_AFFINITY_TYPE_SM_COUNT',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER',
    'CU_FUNC_ATTRIBUTE_BINARY_VERSION',
    'CU_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    'CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 'CU_FUNC_ATTRIBUTE_MAX',
    'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    'CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'CU_FUNC_ATTRIBUTE_NUM_REGS',
    'CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    'CU_FUNC_ATTRIBUTE_PTX_VERSION',
    'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    'CU_FUNC_CACHE_PREFER_EQUAL', 'CU_FUNC_CACHE_PREFER_L1',
    'CU_FUNC_CACHE_PREFER_NONE', 'CU_FUNC_CACHE_PREFER_SHARED',
    'CU_GET_PROC_ADDRESS_DEFAULT',
    'CU_GET_PROC_ADDRESS_LEGACY_STREAM',
    'CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD',
    'CU_GRAPHICS_REGISTER_FLAGS_NONE',
    'CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY',
    'CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST',
    'CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER',
    'CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE', 'CU_GRAPH_EXEC_UPDATE_ERROR',
    'CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE',
    'CU_GRAPH_EXEC_UPDATE_SUCCESS',
    'CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT',
    'CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH',
    'CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT',
    'CU_GRAPH_MEM_ATTR_USED_MEM_HIGH', 'CU_GRAPH_NODE_TYPE_EMPTY',
    'CU_GRAPH_NODE_TYPE_EVENT_RECORD',
    'CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL',
    'CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT', 'CU_GRAPH_NODE_TYPE_GRAPH',
    'CU_GRAPH_NODE_TYPE_HOST', 'CU_GRAPH_NODE_TYPE_KERNEL',
    'CU_GRAPH_NODE_TYPE_MEMCPY', 'CU_GRAPH_NODE_TYPE_MEMSET',
    'CU_GRAPH_NODE_TYPE_MEM_ALLOC', 'CU_GRAPH_NODE_TYPE_MEM_FREE',
    'CU_GRAPH_NODE_TYPE_WAIT_EVENT', 'CU_GRAPH_USER_OBJECT_MOVE',
    'CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS', 'CU_JIT_CACHE_MODE',
    'CU_JIT_CACHE_OPTION_CA', 'CU_JIT_CACHE_OPTION_CG',
    'CU_JIT_CACHE_OPTION_NONE', 'CU_JIT_ERROR_LOG_BUFFER',
    'CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES', 'CU_JIT_FALLBACK_STRATEGY',
    'CU_JIT_FAST_COMPILE', 'CU_JIT_FMA', 'CU_JIT_FTZ',
    'CU_JIT_GENERATE_DEBUG_INFO', 'CU_JIT_GENERATE_LINE_INFO',
    'CU_JIT_GLOBAL_SYMBOL_ADDRESSES', 'CU_JIT_GLOBAL_SYMBOL_COUNT',
    'CU_JIT_GLOBAL_SYMBOL_NAMES', 'CU_JIT_INFO_LOG_BUFFER',
    'CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 'CU_JIT_INPUT_CUBIN',
    'CU_JIT_INPUT_FATBINARY', 'CU_JIT_INPUT_LIBRARY',
    'CU_JIT_INPUT_NVVM', 'CU_JIT_INPUT_OBJECT', 'CU_JIT_INPUT_PTX',
    'CU_JIT_LOG_VERBOSE', 'CU_JIT_LTO', 'CU_JIT_MAX_REGISTERS',
    'CU_JIT_NEW_SM3X_OPT', 'CU_JIT_NUM_INPUT_TYPES',
    'CU_JIT_NUM_OPTIONS', 'CU_JIT_OPTIMIZATION_LEVEL',
    'CU_JIT_PREC_DIV', 'CU_JIT_PREC_SQRT', 'CU_JIT_TARGET',
    'CU_JIT_TARGET_FROM_CUCONTEXT', 'CU_JIT_THREADS_PER_BLOCK',
    'CU_JIT_WALL_TIME',
    'CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    'CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE',
    'CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT',
    'CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH', 'CU_LIMIT_MALLOC_HEAP_SIZE',
    'CU_LIMIT_MAX', 'CU_LIMIT_MAX_L2_FETCH_GRANULARITY',
    'CU_LIMIT_PERSISTING_L2_CACHE_SIZE', 'CU_LIMIT_PRINTF_FIFO_SIZE',
    'CU_LIMIT_STACK_SIZE', 'CU_MEMORYTYPE_ARRAY',
    'CU_MEMORYTYPE_DEVICE', 'CU_MEMORYTYPE_HOST',
    'CU_MEMORYTYPE_UNIFIED', 'CU_MEMPOOL_ATTR_RELEASE_THRESHOLD',
    'CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT',
    'CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH',
    'CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES',
    'CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC',
    'CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES',
    'CU_MEMPOOL_ATTR_USED_MEM_CURRENT',
    'CU_MEMPOOL_ATTR_USED_MEM_HIGH', 'CU_MEM_ACCESS_FLAGS_PROT_MAX',
    'CU_MEM_ACCESS_FLAGS_PROT_NONE', 'CU_MEM_ACCESS_FLAGS_PROT_READ',
    'CU_MEM_ACCESS_FLAGS_PROT_READWRITE',
    'CU_MEM_ADVISE_SET_ACCESSED_BY',
    'CU_MEM_ADVISE_SET_PREFERRED_LOCATION',
    'CU_MEM_ADVISE_SET_READ_MOSTLY',
    'CU_MEM_ADVISE_UNSET_ACCESSED_BY',
    'CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION',
    'CU_MEM_ADVISE_UNSET_READ_MOSTLY',
    'CU_MEM_ALLOCATION_COMP_GENERIC', 'CU_MEM_ALLOCATION_COMP_NONE',
    'CU_MEM_ALLOCATION_TYPE_INVALID', 'CU_MEM_ALLOCATION_TYPE_MAX',
    'CU_MEM_ALLOCATION_TYPE_PINNED',
    'CU_MEM_ALLOC_GRANULARITY_MINIMUM',
    'CU_MEM_ALLOC_GRANULARITY_RECOMMENDED', 'CU_MEM_ATTACH_GLOBAL',
    'CU_MEM_ATTACH_HOST', 'CU_MEM_ATTACH_SINGLE',
    'CU_MEM_HANDLE_TYPE_GENERIC', 'CU_MEM_HANDLE_TYPE_MAX',
    'CU_MEM_HANDLE_TYPE_NONE',
    'CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR',
    'CU_MEM_HANDLE_TYPE_WIN32', 'CU_MEM_HANDLE_TYPE_WIN32_KMT',
    'CU_MEM_LOCATION_TYPE_DEVICE', 'CU_MEM_LOCATION_TYPE_INVALID',
    'CU_MEM_LOCATION_TYPE_MAX', 'CU_MEM_OPERATION_TYPE_MAP',
    'CU_MEM_OPERATION_TYPE_UNMAP',
    'CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY',
    'CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION',
    'CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION',
    'CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY', 'CU_OCCUPANCY_DEFAULT',
    'CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE',
    'CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    'CU_POINTER_ATTRIBUTE_BUFFER_ID', 'CU_POINTER_ATTRIBUTE_CONTEXT',
    'CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    'CU_POINTER_ATTRIBUTE_DEVICE_POINTER',
    'CU_POINTER_ATTRIBUTE_HOST_POINTER',
    'CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    'CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE',
    'CU_POINTER_ATTRIBUTE_IS_MANAGED', 'CU_POINTER_ATTRIBUTE_MAPPED',
    'CU_POINTER_ATTRIBUTE_MEMORY_TYPE',
    'CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
    'CU_POINTER_ATTRIBUTE_P2P_TOKENS',
    'CU_POINTER_ATTRIBUTE_RANGE_SIZE',
    'CU_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    'CU_POINTER_ATTRIBUTE_SYNC_MEMOPS', 'CU_PREFER_BINARY',
    'CU_PREFER_PTX', 'CU_RESOURCE_TYPE_ARRAY',
    'CU_RESOURCE_TYPE_LINEAR', 'CU_RESOURCE_TYPE_MIPMAPPED_ARRAY',
    'CU_RESOURCE_TYPE_PITCH2D', 'CU_RES_VIEW_FORMAT_FLOAT_1X16',
    'CU_RES_VIEW_FORMAT_FLOAT_1X32', 'CU_RES_VIEW_FORMAT_FLOAT_2X16',
    'CU_RES_VIEW_FORMAT_FLOAT_2X32', 'CU_RES_VIEW_FORMAT_FLOAT_4X16',
    'CU_RES_VIEW_FORMAT_FLOAT_4X32', 'CU_RES_VIEW_FORMAT_NONE',
    'CU_RES_VIEW_FORMAT_SIGNED_BC4', 'CU_RES_VIEW_FORMAT_SIGNED_BC5',
    'CU_RES_VIEW_FORMAT_SIGNED_BC6H', 'CU_RES_VIEW_FORMAT_SINT_1X16',
    'CU_RES_VIEW_FORMAT_SINT_1X32', 'CU_RES_VIEW_FORMAT_SINT_1X8',
    'CU_RES_VIEW_FORMAT_SINT_2X16', 'CU_RES_VIEW_FORMAT_SINT_2X32',
    'CU_RES_VIEW_FORMAT_SINT_2X8', 'CU_RES_VIEW_FORMAT_SINT_4X16',
    'CU_RES_VIEW_FORMAT_SINT_4X32', 'CU_RES_VIEW_FORMAT_SINT_4X8',
    'CU_RES_VIEW_FORMAT_UINT_1X16', 'CU_RES_VIEW_FORMAT_UINT_1X32',
    'CU_RES_VIEW_FORMAT_UINT_1X8', 'CU_RES_VIEW_FORMAT_UINT_2X16',
    'CU_RES_VIEW_FORMAT_UINT_2X32', 'CU_RES_VIEW_FORMAT_UINT_2X8',
    'CU_RES_VIEW_FORMAT_UINT_4X16', 'CU_RES_VIEW_FORMAT_UINT_4X32',
    'CU_RES_VIEW_FORMAT_UINT_4X8', 'CU_RES_VIEW_FORMAT_UNSIGNED_BC1',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC2',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC3',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC4',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC5',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC7',
    'CU_SHAREDMEM_CARVEOUT_DEFAULT', 'CU_SHAREDMEM_CARVEOUT_MAX_L1',
    'CU_SHAREDMEM_CARVEOUT_MAX_SHARED',
    'CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE',
    'CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE',
    'CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE',
    'CU_STREAM_ADD_CAPTURE_DEPENDENCIES',
    'CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    'CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY',
    'CU_STREAM_CAPTURE_MODE_GLOBAL', 'CU_STREAM_CAPTURE_MODE_RELAXED',
    'CU_STREAM_CAPTURE_MODE_THREAD_LOCAL',
    'CU_STREAM_CAPTURE_STATUS_ACTIVE',
    'CU_STREAM_CAPTURE_STATUS_INVALIDATED',
    'CU_STREAM_CAPTURE_STATUS_NONE', 'CU_STREAM_DEFAULT',
    'CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES',
    'CU_STREAM_MEM_OP_WAIT_VALUE_32',
    'CU_STREAM_MEM_OP_WAIT_VALUE_64',
    'CU_STREAM_MEM_OP_WRITE_VALUE_32',
    'CU_STREAM_MEM_OP_WRITE_VALUE_64', 'CU_STREAM_NON_BLOCKING',
    'CU_STREAM_SET_CAPTURE_DEPENDENCIES', 'CU_STREAM_WAIT_VALUE_AND',
    'CU_STREAM_WAIT_VALUE_EQ', 'CU_STREAM_WAIT_VALUE_FLUSH',
    'CU_STREAM_WAIT_VALUE_GEQ', 'CU_STREAM_WAIT_VALUE_NOR',
    'CU_STREAM_WRITE_VALUE_DEFAULT',
    'CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER', 'CU_SYNC_POLICY_AUTO',
    'CU_SYNC_POLICY_BLOCKING_SYNC', 'CU_SYNC_POLICY_SPIN',
    'CU_SYNC_POLICY_YIELD', 'CU_TARGET_COMPUTE_20',
    'CU_TARGET_COMPUTE_21', 'CU_TARGET_COMPUTE_30',
    'CU_TARGET_COMPUTE_32', 'CU_TARGET_COMPUTE_35',
    'CU_TARGET_COMPUTE_37', 'CU_TARGET_COMPUTE_50',
    'CU_TARGET_COMPUTE_52', 'CU_TARGET_COMPUTE_53',
    'CU_TARGET_COMPUTE_60', 'CU_TARGET_COMPUTE_61',
    'CU_TARGET_COMPUTE_62', 'CU_TARGET_COMPUTE_70',
    'CU_TARGET_COMPUTE_72', 'CU_TARGET_COMPUTE_75',
    'CU_TARGET_COMPUTE_80', 'CU_TARGET_COMPUTE_86',
    'CU_TR_ADDRESS_MODE_BORDER', 'CU_TR_ADDRESS_MODE_CLAMP',
    'CU_TR_ADDRESS_MODE_MIRROR', 'CU_TR_ADDRESS_MODE_WRAP',
    'CU_TR_FILTER_MODE_LINEAR', 'CU_TR_FILTER_MODE_POINT',
    'CU_USER_OBJECT_NO_DESTRUCTOR_SYNC', 'CUaccessPolicyWindow',
    'CUaccessPolicyWindow_v1', 'CUaccessProperty',
    'CUaccessProperty__enumvalues', 'CUaccessProperty_enum',
    'CUaddress_mode', 'CUaddress_mode__enumvalues',
    'CUaddress_mode_enum', 'CUarray', 'CUarrayMapInfo',
    'CUarrayMapInfo_v1', 'CUarraySparseSubresourceType',
    'CUarraySparseSubresourceType__enumvalues',
    'CUarraySparseSubresourceType_enum', 'CUarray_cubemap_face',
    'CUarray_cubemap_face__enumvalues', 'CUarray_cubemap_face_enum',
    'CUarray_format', 'CUarray_format__enumvalues',
    'CUarray_format_enum', 'CUcomputemode',
    'CUcomputemode__enumvalues', 'CUcomputemode_enum', 'CUcontext',
    'CUctx_flags', 'CUctx_flags__enumvalues', 'CUctx_flags_enum',
    'CUdevice', 'CUdevice_P2PAttribute',
    'CUdevice_P2PAttribute__enumvalues', 'CUdevice_P2PAttribute_enum',
    'CUdevice_attribute', 'CUdevice_attribute__enumvalues',
    'CUdevice_attribute_enum', 'CUdevice_v1', 'CUdeviceptr',
    'CUdeviceptr_v2', 'CUdevprop', 'CUdevprop_v1',
    'CUdriverProcAddress_flags',
    'CUdriverProcAddress_flags__enumvalues',
    'CUdriverProcAddress_flags_enum', 'CUevent', 'CUevent_flags',
    'CUevent_flags__enumvalues', 'CUevent_flags_enum',
    'CUevent_record_flags', 'CUevent_record_flags__enumvalues',
    'CUevent_record_flags_enum', 'CUevent_wait_flags',
    'CUevent_wait_flags__enumvalues', 'CUevent_wait_flags_enum',
    'CUexecAffinityParam', 'CUexecAffinityParam_v1',
    'CUexecAffinitySmCount', 'CUexecAffinitySmCount_v1',
    'CUexecAffinityType', 'CUexecAffinityType__enumvalues',
    'CUexecAffinityType_enum', 'CUexternalMemory',
    'CUexternalMemoryHandleType',
    'CUexternalMemoryHandleType__enumvalues',
    'CUexternalMemoryHandleType_enum', 'CUexternalSemaphore',
    'CUexternalSemaphoreHandleType',
    'CUexternalSemaphoreHandleType__enumvalues',
    'CUexternalSemaphoreHandleType_enum', 'CUfilter_mode',
    'CUfilter_mode__enumvalues', 'CUfilter_mode_enum',
    'CUflushGPUDirectRDMAWritesOptions',
    'CUflushGPUDirectRDMAWritesOptions__enumvalues',
    'CUflushGPUDirectRDMAWritesOptions_enum',
    'CUflushGPUDirectRDMAWritesScope',
    'CUflushGPUDirectRDMAWritesScope__enumvalues',
    'CUflushGPUDirectRDMAWritesScope_enum',
    'CUflushGPUDirectRDMAWritesTarget',
    'CUflushGPUDirectRDMAWritesTarget__enumvalues',
    'CUflushGPUDirectRDMAWritesTarget_enum', 'CUfunc_cache',
    'CUfunc_cache__enumvalues', 'CUfunc_cache_enum', 'CUfunction',
    'CUfunction_attribute', 'CUfunction_attribute__enumvalues',
    'CUfunction_attribute_enum', 'CUgraph', 'CUgraphDebugDot_flags',
    'CUgraphDebugDot_flags__enumvalues', 'CUgraphDebugDot_flags_enum',
    'CUgraphExec', 'CUgraphExecUpdateResult',
    'CUgraphExecUpdateResult__enumvalues',
    'CUgraphExecUpdateResult_enum', 'CUgraphInstantiate_flags',
    'CUgraphInstantiate_flags__enumvalues',
    'CUgraphInstantiate_flags_enum', 'CUgraphMem_attribute',
    'CUgraphMem_attribute__enumvalues', 'CUgraphMem_attribute_enum',
    'CUgraphNode', 'CUgraphNodeType', 'CUgraphNodeType__enumvalues',
    'CUgraphNodeType_enum', 'CUgraphicsMapResourceFlags',
    'CUgraphicsMapResourceFlags__enumvalues',
    'CUgraphicsMapResourceFlags_enum', 'CUgraphicsRegisterFlags',
    'CUgraphicsRegisterFlags__enumvalues',
    'CUgraphicsRegisterFlags_enum', 'CUgraphicsResource', 'CUhostFn',
    'CUipcEventHandle', 'CUipcEventHandle_v1', 'CUipcMemHandle',
    'CUipcMemHandle_v1', 'CUipcMem_flags',
    'CUipcMem_flags__enumvalues', 'CUipcMem_flags_enum',
    'CUjitInputType', 'CUjitInputType__enumvalues',
    'CUjitInputType_enum', 'CUjit_cacheMode',
    'CUjit_cacheMode__enumvalues', 'CUjit_cacheMode_enum',
    'CUjit_fallback', 'CUjit_fallback__enumvalues',
    'CUjit_fallback_enum', 'CUjit_option', 'CUjit_option__enumvalues',
    'CUjit_option_enum', 'CUjit_target', 'CUjit_target__enumvalues',
    'CUjit_target_enum', 'CUkernelNodeAttrID',
    'CUkernelNodeAttrID__enumvalues', 'CUkernelNodeAttrID_enum',
    'CUkernelNodeAttrValue', 'CUkernelNodeAttrValue_v1', 'CUlimit',
    'CUlimit__enumvalues', 'CUlimit_enum', 'CUlinkState',
    'CUmemAccessDesc', 'CUmemAccessDesc_v1', 'CUmemAccess_flags',
    'CUmemAccess_flags__enumvalues', 'CUmemAccess_flags_enum',
    'CUmemAllocationCompType', 'CUmemAllocationCompType__enumvalues',
    'CUmemAllocationCompType_enum',
    'CUmemAllocationGranularity_flags',
    'CUmemAllocationGranularity_flags__enumvalues',
    'CUmemAllocationGranularity_flags_enum',
    'CUmemAllocationHandleType',
    'CUmemAllocationHandleType__enumvalues',
    'CUmemAllocationHandleType_enum', 'CUmemAllocationProp',
    'CUmemAllocationProp_v1', 'CUmemAllocationType',
    'CUmemAllocationType__enumvalues', 'CUmemAllocationType_enum',
    'CUmemAttach_flags', 'CUmemAttach_flags__enumvalues',
    'CUmemAttach_flags_enum', 'CUmemGenericAllocationHandle',
    'CUmemGenericAllocationHandle_v1', 'CUmemHandleType',
    'CUmemHandleType__enumvalues', 'CUmemHandleType_enum',
    'CUmemLocation', 'CUmemLocationType',
    'CUmemLocationType__enumvalues', 'CUmemLocationType_enum',
    'CUmemLocation_v1', 'CUmemOperationType',
    'CUmemOperationType__enumvalues', 'CUmemOperationType_enum',
    'CUmemPoolProps', 'CUmemPoolProps_v1', 'CUmemPoolPtrExportData',
    'CUmemPoolPtrExportData_v1', 'CUmemPool_attribute',
    'CUmemPool_attribute__enumvalues', 'CUmemPool_attribute_enum',
    'CUmem_advise', 'CUmem_advise__enumvalues', 'CUmem_advise_enum',
    'CUmem_range_attribute', 'CUmem_range_attribute__enumvalues',
    'CUmem_range_attribute_enum', 'CUmemoryPool', 'CUmemorytype',
    'CUmemorytype__enumvalues', 'CUmemorytype_enum',
    'CUmipmappedArray', 'CUmodule', 'CUoccupancyB2DSize',
    'CUoccupancy_flags', 'CUoccupancy_flags__enumvalues',
    'CUoccupancy_flags_enum', 'CUpointer_attribute',
    'CUpointer_attribute__enumvalues', 'CUpointer_attribute_enum',
    'CUresourceViewFormat', 'CUresourceViewFormat__enumvalues',
    'CUresourceViewFormat_enum', 'CUresourcetype',
    'CUresourcetype__enumvalues', 'CUresourcetype_enum', 'CUresult',
    'CUresult__enumvalues', 'CUshared_carveout',
    'CUshared_carveout__enumvalues', 'CUshared_carveout_enum',
    'CUsharedconfig', 'CUsharedconfig__enumvalues',
    'CUsharedconfig_enum', 'CUstream', 'CUstreamAttrID',
    'CUstreamAttrID__enumvalues', 'CUstreamAttrID_enum',
    'CUstreamAttrValue', 'CUstreamAttrValue_v1',
    'CUstreamBatchMemOpParams', 'CUstreamBatchMemOpParams_v1',
    'CUstreamBatchMemOpType', 'CUstreamBatchMemOpType__enumvalues',
    'CUstreamBatchMemOpType_enum', 'CUstreamCallback',
    'CUstreamCaptureMode', 'CUstreamCaptureMode__enumvalues',
    'CUstreamCaptureMode_enum', 'CUstreamCaptureStatus',
    'CUstreamCaptureStatus__enumvalues', 'CUstreamCaptureStatus_enum',
    'CUstreamUpdateCaptureDependencies_flags',
    'CUstreamUpdateCaptureDependencies_flags__enumvalues',
    'CUstreamUpdateCaptureDependencies_flags_enum',
    'CUstreamWaitValue_flags', 'CUstreamWaitValue_flags__enumvalues',
    'CUstreamWaitValue_flags_enum', 'CUstreamWriteValue_flags',
    'CUstreamWriteValue_flags__enumvalues',
    'CUstreamWriteValue_flags_enum', 'CUstream_flags',
    'CUstream_flags__enumvalues', 'CUstream_flags_enum',
    'CUsurfObject', 'CUsurfObject_v1', 'CUsurfref',
    'CUsynchronizationPolicy', 'CUsynchronizationPolicy__enumvalues',
    'CUsynchronizationPolicy_enum', 'CUtexObject', 'CUtexObject_v1',
    'CUtexref', 'CUuserObject', 'CUuserObjectRetain_flags',
    'CUuserObjectRetain_flags__enumvalues',
    'CUuserObjectRetain_flags_enum', 'CUuserObject_flags',
    'CUuserObject_flags__enumvalues', 'CUuserObject_flags_enum',
    'CUuuid', 'NVRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    'NVRTC_ERROR_COMPILATION', 'NVRTC_ERROR_INTERNAL_ERROR',
    'NVRTC_ERROR_INVALID_INPUT', 'NVRTC_ERROR_INVALID_OPTION',
    'NVRTC_ERROR_INVALID_PROGRAM',
    'NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    'NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    'NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    'NVRTC_ERROR_OUT_OF_MEMORY',
    'NVRTC_ERROR_PROGRAM_CREATION_FAILURE', 'NVRTC_SUCCESS',
    'c__EA_nvrtcResult', 'cuArray3DCreate_v2',
    'cuArray3DGetDescriptor_v2', 'cuArrayCreate_v2', 'cuArrayDestroy',
    'cuArrayGetDescriptor_v2', 'cuArrayGetPlane',
    'cuArrayGetSparseProperties', 'cuCtxAttach', 'cuCtxCreate_v2',
    'cuCtxCreate_v3', 'cuCtxDestroy_v2', 'cuCtxDetach',
    'cuCtxDisablePeerAccess', 'cuCtxEnablePeerAccess',
    'cuCtxGetApiVersion', 'cuCtxGetCacheConfig', 'cuCtxGetCurrent',
    'cuCtxGetDevice', 'cuCtxGetExecAffinity', 'cuCtxGetFlags',
    'cuCtxGetLimit', 'cuCtxGetSharedMemConfig',
    'cuCtxGetStreamPriorityRange', 'cuCtxPopCurrent_v2',
    'cuCtxPushCurrent_v2', 'cuCtxResetPersistingL2Cache',
    'cuCtxSetCacheConfig', 'cuCtxSetCurrent', 'cuCtxSetLimit',
    'cuCtxSetSharedMemConfig', 'cuCtxSynchronize',
    'cuDestroyExternalMemory', 'cuDestroyExternalSemaphore',
    'cuDeviceCanAccessPeer', 'cuDeviceComputeCapability',
    'cuDeviceGet', 'cuDeviceGetAttribute', 'cuDeviceGetByPCIBusId',
    'cuDeviceGetCount', 'cuDeviceGetDefaultMemPool',
    'cuDeviceGetExecAffinitySupport', 'cuDeviceGetGraphMemAttribute',
    'cuDeviceGetLuid', 'cuDeviceGetMemPool', 'cuDeviceGetName',
    'cuDeviceGetNvSciSyncAttributes', 'cuDeviceGetP2PAttribute',
    'cuDeviceGetPCIBusId', 'cuDeviceGetProperties',
    'cuDeviceGetTexture1DLinearMaxWidth', 'cuDeviceGetUuid',
    'cuDeviceGetUuid_v2', 'cuDeviceGraphMemTrim',
    'cuDevicePrimaryCtxGetState', 'cuDevicePrimaryCtxRelease_v2',
    'cuDevicePrimaryCtxReset_v2', 'cuDevicePrimaryCtxRetain',
    'cuDevicePrimaryCtxSetFlags_v2', 'cuDeviceSetGraphMemAttribute',
    'cuDeviceSetMemPool', 'cuDeviceTotalMem_v2', 'cuDriverGetVersion',
    'cuEventCreate', 'cuEventDestroy_v2', 'cuEventElapsedTime',
    'cuEventQuery', 'cuEventRecord', 'cuEventRecordWithFlags',
    'cuEventSynchronize', 'cuExternalMemoryGetMappedBuffer',
    'cuExternalMemoryGetMappedMipmappedArray',
    'cuFlushGPUDirectRDMAWrites', 'cuFuncGetAttribute',
    'cuFuncGetModule', 'cuFuncSetAttribute', 'cuFuncSetBlockShape',
    'cuFuncSetCacheConfig', 'cuFuncSetSharedMemConfig',
    'cuFuncSetSharedSize', 'cuGetErrorName', 'cuGetErrorString',
    'cuGetExportTable', 'cuGetProcAddress',
    'cuGraphAddChildGraphNode', 'cuGraphAddDependencies',
    'cuGraphAddEmptyNode', 'cuGraphAddEventRecordNode',
    'cuGraphAddEventWaitNode',
    'cuGraphAddExternalSemaphoresSignalNode',
    'cuGraphAddExternalSemaphoresWaitNode', 'cuGraphAddHostNode',
    'cuGraphAddKernelNode', 'cuGraphAddMemAllocNode',
    'cuGraphAddMemFreeNode', 'cuGraphAddMemcpyNode',
    'cuGraphAddMemsetNode', 'cuGraphChildGraphNodeGetGraph',
    'cuGraphClone', 'cuGraphCreate', 'cuGraphDebugDotPrint',
    'cuGraphDestroy', 'cuGraphDestroyNode',
    'cuGraphEventRecordNodeGetEvent',
    'cuGraphEventRecordNodeSetEvent', 'cuGraphEventWaitNodeGetEvent',
    'cuGraphEventWaitNodeSetEvent',
    'cuGraphExecChildGraphNodeSetParams', 'cuGraphExecDestroy',
    'cuGraphExecEventRecordNodeSetEvent',
    'cuGraphExecEventWaitNodeSetEvent',
    'cuGraphExecExternalSemaphoresSignalNodeSetParams',
    'cuGraphExecExternalSemaphoresWaitNodeSetParams',
    'cuGraphExecHostNodeSetParams', 'cuGraphExecKernelNodeSetParams',
    'cuGraphExecMemcpyNodeSetParams',
    'cuGraphExecMemsetNodeSetParams', 'cuGraphExecUpdate',
    'cuGraphExternalSemaphoresSignalNodeGetParams',
    'cuGraphExternalSemaphoresSignalNodeSetParams',
    'cuGraphExternalSemaphoresWaitNodeGetParams',
    'cuGraphExternalSemaphoresWaitNodeSetParams', 'cuGraphGetEdges',
    'cuGraphGetNodes', 'cuGraphGetRootNodes',
    'cuGraphHostNodeGetParams', 'cuGraphHostNodeSetParams',
    'cuGraphInstantiateWithFlags', 'cuGraphInstantiate_v2',
    'cuGraphKernelNodeCopyAttributes',
    'cuGraphKernelNodeGetAttribute', 'cuGraphKernelNodeGetParams',
    'cuGraphKernelNodeSetAttribute', 'cuGraphKernelNodeSetParams',
    'cuGraphLaunch', 'cuGraphMemAllocNodeGetParams',
    'cuGraphMemFreeNodeGetParams', 'cuGraphMemcpyNodeGetParams',
    'cuGraphMemcpyNodeSetParams', 'cuGraphMemsetNodeGetParams',
    'cuGraphMemsetNodeSetParams', 'cuGraphNodeFindInClone',
    'cuGraphNodeGetDependencies', 'cuGraphNodeGetDependentNodes',
    'cuGraphNodeGetType', 'cuGraphReleaseUserObject',
    'cuGraphRemoveDependencies', 'cuGraphRetainUserObject',
    'cuGraphUpload', 'cuGraphicsMapResources',
    'cuGraphicsResourceGetMappedMipmappedArray',
    'cuGraphicsResourceGetMappedPointer_v2',
    'cuGraphicsResourceSetMapFlags_v2',
    'cuGraphicsSubResourceGetMappedArray', 'cuGraphicsUnmapResources',
    'cuGraphicsUnregisterResource', 'cuImportExternalMemory',
    'cuImportExternalSemaphore', 'cuInit', 'cuIpcCloseMemHandle',
    'cuIpcGetEventHandle', 'cuIpcGetMemHandle',
    'cuIpcOpenEventHandle', 'cuIpcOpenMemHandle_v2', 'cuLaunch',
    'cuLaunchCooperativeKernel',
    'cuLaunchCooperativeKernelMultiDevice', 'cuLaunchGrid',
    'cuLaunchGridAsync', 'cuLaunchHostFunc', 'cuLaunchKernel',
    'cuLinkAddData_v2', 'cuLinkAddFile_v2', 'cuLinkComplete',
    'cuLinkCreate_v2', 'cuLinkDestroy', 'cuMemAddressFree',
    'cuMemAddressReserve', 'cuMemAdvise', 'cuMemAllocAsync',
    'cuMemAllocFromPoolAsync', 'cuMemAllocHost_v2',
    'cuMemAllocManaged', 'cuMemAllocPitch_v2', 'cuMemAlloc_v2',
    'cuMemCreate', 'cuMemExportToShareableHandle', 'cuMemFreeAsync',
    'cuMemFreeHost', 'cuMemFree_v2', 'cuMemGetAccess',
    'cuMemGetAddressRange_v2', 'cuMemGetAllocationGranularity',
    'cuMemGetAllocationPropertiesFromHandle', 'cuMemGetInfo_v2',
    'cuMemHostAlloc', 'cuMemHostGetDevicePointer_v2',
    'cuMemHostGetFlags', 'cuMemHostRegister_v2',
    'cuMemHostUnregister', 'cuMemImportFromShareableHandle',
    'cuMemMap', 'cuMemMapArrayAsync', 'cuMemPoolCreate',
    'cuMemPoolDestroy', 'cuMemPoolExportPointer',
    'cuMemPoolExportToShareableHandle', 'cuMemPoolGetAccess',
    'cuMemPoolGetAttribute', 'cuMemPoolImportFromShareableHandle',
    'cuMemPoolImportPointer', 'cuMemPoolSetAccess',
    'cuMemPoolSetAttribute', 'cuMemPoolTrimTo', 'cuMemPrefetchAsync',
    'cuMemRangeGetAttribute', 'cuMemRangeGetAttributes',
    'cuMemRelease', 'cuMemRetainAllocationHandle', 'cuMemSetAccess',
    'cuMemUnmap', 'cuMemcpy', 'cuMemcpy2DAsync_v2',
    'cuMemcpy2DUnaligned_v2', 'cuMemcpy2D_v2', 'cuMemcpy3DAsync_v2',
    'cuMemcpy3DPeer', 'cuMemcpy3DPeerAsync', 'cuMemcpy3D_v2',
    'cuMemcpyAsync', 'cuMemcpyAtoA_v2', 'cuMemcpyAtoD_v2',
    'cuMemcpyAtoHAsync_v2', 'cuMemcpyAtoH_v2', 'cuMemcpyDtoA_v2',
    'cuMemcpyDtoDAsync_v2', 'cuMemcpyDtoD_v2', 'cuMemcpyDtoHAsync_v2',
    'cuMemcpyDtoH_v2', 'cuMemcpyHtoAAsync_v2', 'cuMemcpyHtoA_v2',
    'cuMemcpyHtoDAsync_v2', 'cuMemcpyHtoD_v2', 'cuMemcpyPeer',
    'cuMemcpyPeerAsync', 'cuMemsetD16Async', 'cuMemsetD16_v2',
    'cuMemsetD2D16Async', 'cuMemsetD2D16_v2', 'cuMemsetD2D32Async',
    'cuMemsetD2D32_v2', 'cuMemsetD2D8Async', 'cuMemsetD2D8_v2',
    'cuMemsetD32Async', 'cuMemsetD32_v2', 'cuMemsetD8Async',
    'cuMemsetD8_v2', 'cuMipmappedArrayCreate',
    'cuMipmappedArrayDestroy', 'cuMipmappedArrayGetLevel',
    'cuMipmappedArrayGetSparseProperties', 'cuModuleGetFunction',
    'cuModuleGetGlobal_v2', 'cuModuleGetSurfRef', 'cuModuleGetTexRef',
    'cuModuleLoad', 'cuModuleLoadData', 'cuModuleLoadDataEx',
    'cuModuleLoadFatBinary', 'cuModuleUnload',
    'cuOccupancyAvailableDynamicSMemPerBlock',
    'cuOccupancyMaxActiveBlocksPerMultiprocessor',
    'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags',
    'cuOccupancyMaxPotentialBlockSize',
    'cuOccupancyMaxPotentialBlockSizeWithFlags', 'cuParamSetSize',
    'cuParamSetTexRef', 'cuParamSetf', 'cuParamSeti', 'cuParamSetv',
    'cuPointerGetAttribute', 'cuPointerGetAttributes',
    'cuPointerSetAttribute', 'cuSignalExternalSemaphoresAsync',
    'cuStreamAddCallback', 'cuStreamAttachMemAsync',
    'cuStreamBatchMemOp', 'cuStreamBeginCapture_v2',
    'cuStreamCopyAttributes', 'cuStreamCreate',
    'cuStreamCreateWithPriority', 'cuStreamDestroy_v2',
    'cuStreamEndCapture', 'cuStreamGetAttribute',
    'cuStreamGetCaptureInfo', 'cuStreamGetCaptureInfo_v2',
    'cuStreamGetCtx', 'cuStreamGetFlags', 'cuStreamGetPriority',
    'cuStreamIsCapturing', 'cuStreamQuery', 'cuStreamSetAttribute',
    'cuStreamSynchronize', 'cuStreamUpdateCaptureDependencies',
    'cuStreamWaitEvent', 'cuStreamWaitValue32', 'cuStreamWaitValue64',
    'cuStreamWriteValue32', 'cuStreamWriteValue64',
    'cuSurfObjectCreate', 'cuSurfObjectDestroy',
    'cuSurfObjectGetResourceDesc', 'cuSurfRefGetArray',
    'cuSurfRefSetArray', 'cuTexObjectCreate', 'cuTexObjectDestroy',
    'cuTexObjectGetResourceDesc', 'cuTexObjectGetResourceViewDesc',
    'cuTexObjectGetTextureDesc', 'cuTexRefCreate', 'cuTexRefDestroy',
    'cuTexRefGetAddressMode', 'cuTexRefGetAddress_v2',
    'cuTexRefGetArray', 'cuTexRefGetBorderColor',
    'cuTexRefGetFilterMode', 'cuTexRefGetFlags', 'cuTexRefGetFormat',
    'cuTexRefGetMaxAnisotropy', 'cuTexRefGetMipmapFilterMode',
    'cuTexRefGetMipmapLevelBias', 'cuTexRefGetMipmapLevelClamp',
    'cuTexRefGetMipmappedArray', 'cuTexRefSetAddress2D_v3',
    'cuTexRefSetAddressMode', 'cuTexRefSetAddress_v2',
    'cuTexRefSetArray', 'cuTexRefSetBorderColor',
    'cuTexRefSetFilterMode', 'cuTexRefSetFlags', 'cuTexRefSetFormat',
    'cuTexRefSetMaxAnisotropy', 'cuTexRefSetMipmapFilterMode',
    'cuTexRefSetMipmapLevelBias', 'cuTexRefSetMipmapLevelClamp',
    'cuTexRefSetMipmappedArray', 'cuThreadExchangeStreamCaptureMode',
    'cuUserObjectCreate', 'cuUserObjectRelease', 'cuUserObjectRetain',
    'cuWaitExternalSemaphoresAsync', 'cudaError_enum', 'cuuint32_t',
    'cuuint64_t', 'nvrtcAddNameExpression', 'nvrtcCompileProgram',
    'nvrtcCreateProgram', 'nvrtcDestroyProgram', 'nvrtcGetCUBIN',
    'nvrtcGetCUBINSize', 'nvrtcGetErrorString', 'nvrtcGetLoweredName',
    'nvrtcGetNVVM', 'nvrtcGetNVVMSize', 'nvrtcGetNumSupportedArchs',
    'nvrtcGetPTX', 'nvrtcGetPTXSize', 'nvrtcGetProgramLog',
    'nvrtcGetProgramLogSize', 'nvrtcGetSupportedArchs',
    'nvrtcProgram', 'nvrtcResult', 'nvrtcResult__enumvalues',
    'nvrtcVersion', 'size_t', 'struct_CUDA_ARRAY3D_DESCRIPTOR_st',
    'struct_CUDA_ARRAY_DESCRIPTOR_st',
    'struct_CUDA_ARRAY_SPARSE_PROPERTIES_st',
    'struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent',
    'struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st',
    'struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st',
    'struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32',
    'struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params',
    'struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st',
    'struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st',
    'struct_CUDA_HOST_NODE_PARAMS_st',
    'struct_CUDA_KERNEL_NODE_PARAMS_st',
    'struct_CUDA_LAUNCH_PARAMS_st', 'struct_CUDA_MEMCPY2D_st',
    'struct_CUDA_MEMCPY3D_PEER_st', 'struct_CUDA_MEMCPY3D_st',
    'struct_CUDA_MEMSET_NODE_PARAMS_st',
    'struct_CUDA_MEM_ALLOC_NODE_PARAMS_st',
    'struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st',
    'struct_CUDA_RESOURCE_DESC_st',
    'struct_CUDA_RESOURCE_DESC_st_0_array',
    'struct_CUDA_RESOURCE_DESC_st_0_linear',
    'struct_CUDA_RESOURCE_DESC_st_0_mipmap',
    'struct_CUDA_RESOURCE_DESC_st_0_pitch2D',
    'struct_CUDA_RESOURCE_DESC_st_0_reserved',
    'struct_CUDA_RESOURCE_VIEW_DESC_st',
    'struct_CUDA_TEXTURE_DESC_st', 'struct_CUaccessPolicyWindow_st',
    'struct_CUarrayMapInfo_st', 'struct_CUarrayMapInfo_st_1_miptail',
    'struct_CUarrayMapInfo_st_1_sparseLevel', 'struct_CUarray_st',
    'struct_CUctx_st', 'struct_CUdevprop_st', 'struct_CUevent_st',
    'struct_CUexecAffinityParam_st',
    'struct_CUexecAffinitySmCount_st', 'struct_CUextMemory_st',
    'struct_CUextSemaphore_st', 'struct_CUfunc_st',
    'struct_CUgraphExec_st', 'struct_CUgraphNode_st',
    'struct_CUgraph_st', 'struct_CUgraphicsResource_st',
    'struct_CUipcEventHandle_st', 'struct_CUipcMemHandle_st',
    'struct_CUlinkState_st', 'struct_CUmemAccessDesc_st',
    'struct_CUmemAllocationProp_st',
    'struct_CUmemAllocationProp_st_allocFlags',
    'struct_CUmemLocation_st', 'struct_CUmemPoolHandle_st',
    'struct_CUmemPoolProps_st', 'struct_CUmemPoolPtrExportData_st',
    'struct_CUmipmappedArray_st', 'struct_CUmod_st',
    'struct_CUstreamMemOpFlushRemoteWritesParams_st',
    'struct_CUstreamMemOpWaitValueParams_st',
    'struct_CUstreamMemOpWriteValueParams_st', 'struct_CUstream_st',
    'struct_CUsurfref_st', 'struct_CUtexref_st',
    'struct_CUuserObject_st', 'struct_CUuuid_st',
    'struct__nvrtcProgram',
    'union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle',
    'union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle',
    'union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync',
    'union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync',
    'union_CUDA_RESOURCE_DESC_st_res',
    'union_CUarrayMapInfo_st_memHandle',
    'union_CUarrayMapInfo_st_resource',
    'union_CUarrayMapInfo_st_subresource',
    'union_CUexecAffinityParam_st_param',
    'union_CUkernelNodeAttrValue_union',
    'union_CUstreamAttrValue_union',
    'union_CUstreamBatchMemOpParams_union',
    'union_CUstreamMemOpWaitValueParams_st_0',
    'union_CUstreamMemOpWriteValueParams_st_0']


# tinygrad/runtime/autogen/hip.py

# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-D__HIP_PLATFORM_AMD__', '-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries['libamdhip64.so'] = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')



# values for enumeration 'c__Ea_HIP_SUCCESS'
c__Ea_HIP_SUCCESS__enumvalues = {
    0: 'HIP_SUCCESS',
    1: 'HIP_ERROR_INVALID_VALUE',
    2: 'HIP_ERROR_NOT_INITIALIZED',
    3: 'HIP_ERROR_LAUNCH_OUT_OF_RESOURCES',
}
HIP_SUCCESS = 0
HIP_ERROR_INVALID_VALUE = 1
HIP_ERROR_NOT_INITIALIZED = 2
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = 3
c__Ea_HIP_SUCCESS = ctypes.c_uint32 # enum
class struct_c__SA_hipDeviceArch_t(Structure):
    pass

struct_c__SA_hipDeviceArch_t._pack_ = 1 # source:False
struct_c__SA_hipDeviceArch_t._fields_ = [
    ('hasGlobalInt32Atomics', ctypes.c_uint32, 1),
    ('hasGlobalFloatAtomicExch', ctypes.c_uint32, 1),
    ('hasSharedInt32Atomics', ctypes.c_uint32, 1),
    ('hasSharedFloatAtomicExch', ctypes.c_uint32, 1),
    ('hasFloatAtomicAdd', ctypes.c_uint32, 1),
    ('hasGlobalInt64Atomics', ctypes.c_uint32, 1),
    ('hasSharedInt64Atomics', ctypes.c_uint32, 1),
    ('hasDoubles', ctypes.c_uint32, 1),
    ('hasWarpVote', ctypes.c_uint32, 1),
    ('hasWarpBallot', ctypes.c_uint32, 1),
    ('hasWarpShuffle', ctypes.c_uint32, 1),
    ('hasFunnelShift', ctypes.c_uint32, 1),
    ('hasThreadFenceSystem', ctypes.c_uint32, 1),
    ('hasSyncThreadsExt', ctypes.c_uint32, 1),
    ('hasSurfaceFuncs', ctypes.c_uint32, 1),
    ('has3dGrid', ctypes.c_uint32, 1),
    ('hasDynamicParallelism', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint16, 15),
]

hipDeviceArch_t = struct_c__SA_hipDeviceArch_t
class struct_hipUUID_t(Structure):
    pass

struct_hipUUID_t._pack_ = 1 # source:False
struct_hipUUID_t._fields_ = [
    ('bytes', ctypes.c_char * 16),
]

hipUUID = struct_hipUUID_t
class struct_hipDeviceProp_tR0600(Structure):
    pass

struct_hipDeviceProp_tR0600._pack_ = 1 # source:False
struct_hipDeviceProp_tR0600._fields_ = [
    ('name', ctypes.c_char * 256),
    ('uuid', hipUUID),
    ('luid', ctypes.c_char * 8),
    ('luidDeviceNodeMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('totalGlobalMem', ctypes.c_uint64),
    ('sharedMemPerBlock', ctypes.c_uint64),
    ('regsPerBlock', ctypes.c_int32),
    ('warpSize', ctypes.c_int32),
    ('memPitch', ctypes.c_uint64),
    ('maxThreadsPerBlock', ctypes.c_int32),
    ('maxThreadsDim', ctypes.c_int32 * 3),
    ('maxGridSize', ctypes.c_int32 * 3),
    ('clockRate', ctypes.c_int32),
    ('totalConstMem', ctypes.c_uint64),
    ('major', ctypes.c_int32),
    ('minor', ctypes.c_int32),
    ('textureAlignment', ctypes.c_uint64),
    ('texturePitchAlignment', ctypes.c_uint64),
    ('deviceOverlap', ctypes.c_int32),
    ('multiProcessorCount', ctypes.c_int32),
    ('kernelExecTimeoutEnabled', ctypes.c_int32),
    ('integrated', ctypes.c_int32),
    ('canMapHostMemory', ctypes.c_int32),
    ('computeMode', ctypes.c_int32),
    ('maxTexture1D', ctypes.c_int32),
    ('maxTexture1DMipmap', ctypes.c_int32),
    ('maxTexture1DLinear', ctypes.c_int32),
    ('maxTexture2D', ctypes.c_int32 * 2),
    ('maxTexture2DMipmap', ctypes.c_int32 * 2),
    ('maxTexture2DLinear', ctypes.c_int32 * 3),
    ('maxTexture2DGather', ctypes.c_int32 * 2),
    ('maxTexture3D', ctypes.c_int32 * 3),
    ('maxTexture3DAlt', ctypes.c_int32 * 3),
    ('maxTextureCubemap', ctypes.c_int32),
    ('maxTexture1DLayered', ctypes.c_int32 * 2),
    ('maxTexture2DLayered', ctypes.c_int32 * 3),
    ('maxTextureCubemapLayered', ctypes.c_int32 * 2),
    ('maxSurface1D', ctypes.c_int32),
    ('maxSurface2D', ctypes.c_int32 * 2),
    ('maxSurface3D', ctypes.c_int32 * 3),
    ('maxSurface1DLayered', ctypes.c_int32 * 2),
    ('maxSurface2DLayered', ctypes.c_int32 * 3),
    ('maxSurfaceCubemap', ctypes.c_int32),
    ('maxSurfaceCubemapLayered', ctypes.c_int32 * 2),
    ('surfaceAlignment', ctypes.c_uint64),
    ('concurrentKernels', ctypes.c_int32),
    ('ECCEnabled', ctypes.c_int32),
    ('pciBusID', ctypes.c_int32),
    ('pciDeviceID', ctypes.c_int32),
    ('pciDomainID', ctypes.c_int32),
    ('tccDriver', ctypes.c_int32),
    ('asyncEngineCount', ctypes.c_int32),
    ('unifiedAddressing', ctypes.c_int32),
    ('memoryClockRate', ctypes.c_int32),
    ('memoryBusWidth', ctypes.c_int32),
    ('l2CacheSize', ctypes.c_int32),
    ('persistingL2CacheMaxSize', ctypes.c_int32),
    ('maxThreadsPerMultiProcessor', ctypes.c_int32),
    ('streamPrioritiesSupported', ctypes.c_int32),
    ('globalL1CacheSupported', ctypes.c_int32),
    ('localL1CacheSupported', ctypes.c_int32),
    ('sharedMemPerMultiprocessor', ctypes.c_uint64),
    ('regsPerMultiprocessor', ctypes.c_int32),
    ('managedMemory', ctypes.c_int32),
    ('isMultiGpuBoard', ctypes.c_int32),
    ('multiGpuBoardGroupID', ctypes.c_int32),
    ('hostNativeAtomicSupported', ctypes.c_int32),
    ('singleToDoublePrecisionPerfRatio', ctypes.c_int32),
    ('pageableMemoryAccess', ctypes.c_int32),
    ('concurrentManagedAccess', ctypes.c_int32),
    ('computePreemptionSupported', ctypes.c_int32),
    ('canUseHostPointerForRegisteredMem', ctypes.c_int32),
    ('cooperativeLaunch', ctypes.c_int32),
    ('cooperativeMultiDeviceLaunch', ctypes.c_int32),
    ('sharedMemPerBlockOptin', ctypes.c_uint64),
    ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int32),
    ('directManagedMemAccessFromHost', ctypes.c_int32),
    ('maxBlocksPerMultiProcessor', ctypes.c_int32),
    ('accessPolicyMaxWindowSize', ctypes.c_int32),
    ('reservedSharedMemPerBlock', ctypes.c_uint64),
    ('hostRegisterSupported', ctypes.c_int32),
    ('sparseHipArraySupported', ctypes.c_int32),
    ('hostRegisterReadOnlySupported', ctypes.c_int32),
    ('timelineSemaphoreInteropSupported', ctypes.c_int32),
    ('memoryPoolsSupported', ctypes.c_int32),
    ('gpuDirectRDMASupported', ctypes.c_int32),
    ('gpuDirectRDMAFlushWritesOptions', ctypes.c_uint32),
    ('gpuDirectRDMAWritesOrdering', ctypes.c_int32),
    ('memoryPoolSupportedHandleTypes', ctypes.c_uint32),
    ('deferredMappingHipArraySupported', ctypes.c_int32),
    ('ipcEventSupported', ctypes.c_int32),
    ('clusterLaunch', ctypes.c_int32),
    ('unifiedFunctionPointers', ctypes.c_int32),
    ('reserved', ctypes.c_int32 * 63),
    ('hipReserved', ctypes.c_int32 * 32),
    ('gcnArchName', ctypes.c_char * 256),
    ('maxSharedMemoryPerMultiProcessor', ctypes.c_uint64),
    ('clockInstructionRate', ctypes.c_int32),
    ('arch', hipDeviceArch_t),
    ('hdpMemFlushCntl', ctypes.POINTER(ctypes.c_uint32)),
    ('hdpRegFlushCntl', ctypes.POINTER(ctypes.c_uint32)),
    ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int32),
    ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int32),
    ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int32),
    ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int32),
    ('isLargeBar', ctypes.c_int32),
    ('asicRevision', ctypes.c_int32),
]

hipDeviceProp_tR0600 = struct_hipDeviceProp_tR0600

# values for enumeration 'hipMemoryType'
hipMemoryType__enumvalues = {
    0: 'hipMemoryTypeUnregistered',
    1: 'hipMemoryTypeHost',
    2: 'hipMemoryTypeDevice',
    3: 'hipMemoryTypeManaged',
    10: 'hipMemoryTypeArray',
    11: 'hipMemoryTypeUnified',
}
hipMemoryTypeUnregistered = 0
hipMemoryTypeHost = 1
hipMemoryTypeDevice = 2
hipMemoryTypeManaged = 3
hipMemoryTypeArray = 10
hipMemoryTypeUnified = 11
hipMemoryType = ctypes.c_uint32 # enum
class struct_hipPointerAttribute_t(Structure):
    pass

struct_hipPointerAttribute_t._pack_ = 1 # source:False
struct_hipPointerAttribute_t._fields_ = [
    ('type', hipMemoryType),
    ('device', ctypes.c_int32),
    ('devicePointer', ctypes.POINTER(None)),
    ('hostPointer', ctypes.POINTER(None)),
    ('isManaged', ctypes.c_int32),
    ('allocationFlags', ctypes.c_uint32),
]

hipPointerAttribute_t = struct_hipPointerAttribute_t

# values for enumeration 'hipError_t'
hipError_t__enumvalues = {
    0: 'hipSuccess',
    1: 'hipErrorInvalidValue',
    2: 'hipErrorOutOfMemory',
    2: 'hipErrorMemoryAllocation',
    3: 'hipErrorNotInitialized',
    3: 'hipErrorInitializationError',
    4: 'hipErrorDeinitialized',
    5: 'hipErrorProfilerDisabled',
    6: 'hipErrorProfilerNotInitialized',
    7: 'hipErrorProfilerAlreadyStarted',
    8: 'hipErrorProfilerAlreadyStopped',
    9: 'hipErrorInvalidConfiguration',
    12: 'hipErrorInvalidPitchValue',
    13: 'hipErrorInvalidSymbol',
    17: 'hipErrorInvalidDevicePointer',
    21: 'hipErrorInvalidMemcpyDirection',
    35: 'hipErrorInsufficientDriver',
    52: 'hipErrorMissingConfiguration',
    53: 'hipErrorPriorLaunchFailure',
    98: 'hipErrorInvalidDeviceFunction',
    100: 'hipErrorNoDevice',
    101: 'hipErrorInvalidDevice',
    200: 'hipErrorInvalidImage',
    201: 'hipErrorInvalidContext',
    202: 'hipErrorContextAlreadyCurrent',
    205: 'hipErrorMapFailed',
    205: 'hipErrorMapBufferObjectFailed',
    206: 'hipErrorUnmapFailed',
    207: 'hipErrorArrayIsMapped',
    208: 'hipErrorAlreadyMapped',
    209: 'hipErrorNoBinaryForGpu',
    210: 'hipErrorAlreadyAcquired',
    211: 'hipErrorNotMapped',
    212: 'hipErrorNotMappedAsArray',
    213: 'hipErrorNotMappedAsPointer',
    214: 'hipErrorECCNotCorrectable',
    215: 'hipErrorUnsupportedLimit',
    216: 'hipErrorContextAlreadyInUse',
    217: 'hipErrorPeerAccessUnsupported',
    218: 'hipErrorInvalidKernelFile',
    219: 'hipErrorInvalidGraphicsContext',
    300: 'hipErrorInvalidSource',
    301: 'hipErrorFileNotFound',
    302: 'hipErrorSharedObjectSymbolNotFound',
    303: 'hipErrorSharedObjectInitFailed',
    304: 'hipErrorOperatingSystem',
    400: 'hipErrorInvalidHandle',
    400: 'hipErrorInvalidResourceHandle',
    401: 'hipErrorIllegalState',
    500: 'hipErrorNotFound',
    600: 'hipErrorNotReady',
    700: 'hipErrorIllegalAddress',
    701: 'hipErrorLaunchOutOfResources',
    702: 'hipErrorLaunchTimeOut',
    704: 'hipErrorPeerAccessAlreadyEnabled',
    705: 'hipErrorPeerAccessNotEnabled',
    708: 'hipErrorSetOnActiveProcess',
    709: 'hipErrorContextIsDestroyed',
    710: 'hipErrorAssert',
    712: 'hipErrorHostMemoryAlreadyRegistered',
    713: 'hipErrorHostMemoryNotRegistered',
    719: 'hipErrorLaunchFailure',
    720: 'hipErrorCooperativeLaunchTooLarge',
    801: 'hipErrorNotSupported',
    900: 'hipErrorStreamCaptureUnsupported',
    901: 'hipErrorStreamCaptureInvalidated',
    902: 'hipErrorStreamCaptureMerge',
    903: 'hipErrorStreamCaptureUnmatched',
    904: 'hipErrorStreamCaptureUnjoined',
    905: 'hipErrorStreamCaptureIsolation',
    906: 'hipErrorStreamCaptureImplicit',
    907: 'hipErrorCapturedEvent',
    908: 'hipErrorStreamCaptureWrongThread',
    910: 'hipErrorGraphExecUpdateFailure',
    999: 'hipErrorUnknown',
    1052: 'hipErrorRuntimeMemory',
    1053: 'hipErrorRuntimeOther',
    1054: 'hipErrorTbd',
}
hipSuccess = 0
hipErrorInvalidValue = 1
hipErrorOutOfMemory = 2
hipErrorMemoryAllocation = 2
hipErrorNotInitialized = 3
hipErrorInitializationError = 3
hipErrorDeinitialized = 4
hipErrorProfilerDisabled = 5
hipErrorProfilerNotInitialized = 6
hipErrorProfilerAlreadyStarted = 7
hipErrorProfilerAlreadyStopped = 8
hipErrorInvalidConfiguration = 9
hipErrorInvalidPitchValue = 12
hipErrorInvalidSymbol = 13
hipErrorInvalidDevicePointer = 17
hipErrorInvalidMemcpyDirection = 21
hipErrorInsufficientDriver = 35
hipErrorMissingConfiguration = 52
hipErrorPriorLaunchFailure = 53
hipErrorInvalidDeviceFunction = 98
hipErrorNoDevice = 100
hipErrorInvalidDevice = 101
hipErrorInvalidImage = 200
hipErrorInvalidContext = 201
hipErrorContextAlreadyCurrent = 202
hipErrorMapFailed = 205
hipErrorMapBufferObjectFailed = 205
hipErrorUnmapFailed = 206
hipErrorArrayIsMapped = 207
hipErrorAlreadyMapped = 208
hipErrorNoBinaryForGpu = 209
hipErrorAlreadyAcquired = 210
hipErrorNotMapped = 211
hipErrorNotMappedAsArray = 212
hipErrorNotMappedAsPointer = 213
hipErrorECCNotCorrectable = 214
hipErrorUnsupportedLimit = 215
hipErrorContextAlreadyInUse = 216
hipErrorPeerAccessUnsupported = 217
hipErrorInvalidKernelFile = 218
hipErrorInvalidGraphicsContext = 219
hipErrorInvalidSource = 300
hipErrorFileNotFound = 301
hipErrorSharedObjectSymbolNotFound = 302
hipErrorSharedObjectInitFailed = 303
hipErrorOperatingSystem = 304
hipErrorInvalidHandle = 400
hipErrorInvalidResourceHandle = 400
hipErrorIllegalState = 401
hipErrorNotFound = 500
hipErrorNotReady = 600
hipErrorIllegalAddress = 700
hipErrorLaunchOutOfResources = 701
hipErrorLaunchTimeOut = 702
hipErrorPeerAccessAlreadyEnabled = 704
hipErrorPeerAccessNotEnabled = 705
hipErrorSetOnActiveProcess = 708
hipErrorContextIsDestroyed = 709
hipErrorAssert = 710
hipErrorHostMemoryAlreadyRegistered = 712
hipErrorHostMemoryNotRegistered = 713
hipErrorLaunchFailure = 719
hipErrorCooperativeLaunchTooLarge = 720
hipErrorNotSupported = 801
hipErrorStreamCaptureUnsupported = 900
hipErrorStreamCaptureInvalidated = 901
hipErrorStreamCaptureMerge = 902
hipErrorStreamCaptureUnmatched = 903
hipErrorStreamCaptureUnjoined = 904
hipErrorStreamCaptureIsolation = 905
hipErrorStreamCaptureImplicit = 906
hipErrorCapturedEvent = 907
hipErrorStreamCaptureWrongThread = 908
hipErrorGraphExecUpdateFailure = 910
hipErrorUnknown = 999
hipErrorRuntimeMemory = 1052
hipErrorRuntimeOther = 1053
hipErrorTbd = 1054
hipError_t = ctypes.c_uint32 # enum

# values for enumeration 'hipDeviceAttribute_t'
hipDeviceAttribute_t__enumvalues = {
    0: 'hipDeviceAttributeCudaCompatibleBegin',
    0: 'hipDeviceAttributeEccEnabled',
    1: 'hipDeviceAttributeAccessPolicyMaxWindowSize',
    2: 'hipDeviceAttributeAsyncEngineCount',
    3: 'hipDeviceAttributeCanMapHostMemory',
    4: 'hipDeviceAttributeCanUseHostPointerForRegisteredMem',
    5: 'hipDeviceAttributeClockRate',
    6: 'hipDeviceAttributeComputeMode',
    7: 'hipDeviceAttributeComputePreemptionSupported',
    8: 'hipDeviceAttributeConcurrentKernels',
    9: 'hipDeviceAttributeConcurrentManagedAccess',
    10: 'hipDeviceAttributeCooperativeLaunch',
    11: 'hipDeviceAttributeCooperativeMultiDeviceLaunch',
    12: 'hipDeviceAttributeDeviceOverlap',
    13: 'hipDeviceAttributeDirectManagedMemAccessFromHost',
    14: 'hipDeviceAttributeGlobalL1CacheSupported',
    15: 'hipDeviceAttributeHostNativeAtomicSupported',
    16: 'hipDeviceAttributeIntegrated',
    17: 'hipDeviceAttributeIsMultiGpuBoard',
    18: 'hipDeviceAttributeKernelExecTimeout',
    19: 'hipDeviceAttributeL2CacheSize',
    20: 'hipDeviceAttributeLocalL1CacheSupported',
    21: 'hipDeviceAttributeLuid',
    22: 'hipDeviceAttributeLuidDeviceNodeMask',
    23: 'hipDeviceAttributeComputeCapabilityMajor',
    24: 'hipDeviceAttributeManagedMemory',
    25: 'hipDeviceAttributeMaxBlocksPerMultiProcessor',
    26: 'hipDeviceAttributeMaxBlockDimX',
    27: 'hipDeviceAttributeMaxBlockDimY',
    28: 'hipDeviceAttributeMaxBlockDimZ',
    29: 'hipDeviceAttributeMaxGridDimX',
    30: 'hipDeviceAttributeMaxGridDimY',
    31: 'hipDeviceAttributeMaxGridDimZ',
    32: 'hipDeviceAttributeMaxSurface1D',
    33: 'hipDeviceAttributeMaxSurface1DLayered',
    34: 'hipDeviceAttributeMaxSurface2D',
    35: 'hipDeviceAttributeMaxSurface2DLayered',
    36: 'hipDeviceAttributeMaxSurface3D',
    37: 'hipDeviceAttributeMaxSurfaceCubemap',
    38: 'hipDeviceAttributeMaxSurfaceCubemapLayered',
    39: 'hipDeviceAttributeMaxTexture1DWidth',
    40: 'hipDeviceAttributeMaxTexture1DLayered',
    41: 'hipDeviceAttributeMaxTexture1DLinear',
    42: 'hipDeviceAttributeMaxTexture1DMipmap',
    43: 'hipDeviceAttributeMaxTexture2DWidth',
    44: 'hipDeviceAttributeMaxTexture2DHeight',
    45: 'hipDeviceAttributeMaxTexture2DGather',
    46: 'hipDeviceAttributeMaxTexture2DLayered',
    47: 'hipDeviceAttributeMaxTexture2DLinear',
    48: 'hipDeviceAttributeMaxTexture2DMipmap',
    49: 'hipDeviceAttributeMaxTexture3DWidth',
    50: 'hipDeviceAttributeMaxTexture3DHeight',
    51: 'hipDeviceAttributeMaxTexture3DDepth',
    52: 'hipDeviceAttributeMaxTexture3DAlt',
    53: 'hipDeviceAttributeMaxTextureCubemap',
    54: 'hipDeviceAttributeMaxTextureCubemapLayered',
    55: 'hipDeviceAttributeMaxThreadsDim',
    56: 'hipDeviceAttributeMaxThreadsPerBlock',
    57: 'hipDeviceAttributeMaxThreadsPerMultiProcessor',
    58: 'hipDeviceAttributeMaxPitch',
    59: 'hipDeviceAttributeMemoryBusWidth',
    60: 'hipDeviceAttributeMemoryClockRate',
    61: 'hipDeviceAttributeComputeCapabilityMinor',
    62: 'hipDeviceAttributeMultiGpuBoardGroupID',
    63: 'hipDeviceAttributeMultiprocessorCount',
    64: 'hipDeviceAttributeUnused1',
    65: 'hipDeviceAttributePageableMemoryAccess',
    66: 'hipDeviceAttributePageableMemoryAccessUsesHostPageTables',
    67: 'hipDeviceAttributePciBusId',
    68: 'hipDeviceAttributePciDeviceId',
    69: 'hipDeviceAttributePciDomainID',
    70: 'hipDeviceAttributePersistingL2CacheMaxSize',
    71: 'hipDeviceAttributeMaxRegistersPerBlock',
    72: 'hipDeviceAttributeMaxRegistersPerMultiprocessor',
    73: 'hipDeviceAttributeReservedSharedMemPerBlock',
    74: 'hipDeviceAttributeMaxSharedMemoryPerBlock',
    75: 'hipDeviceAttributeSharedMemPerBlockOptin',
    76: 'hipDeviceAttributeSharedMemPerMultiprocessor',
    77: 'hipDeviceAttributeSingleToDoublePrecisionPerfRatio',
    78: 'hipDeviceAttributeStreamPrioritiesSupported',
    79: 'hipDeviceAttributeSurfaceAlignment',
    80: 'hipDeviceAttributeTccDriver',
    81: 'hipDeviceAttributeTextureAlignment',
    82: 'hipDeviceAttributeTexturePitchAlignment',
    83: 'hipDeviceAttributeTotalConstantMemory',
    84: 'hipDeviceAttributeTotalGlobalMem',
    85: 'hipDeviceAttributeUnifiedAddressing',
    86: 'hipDeviceAttributeUnused2',
    87: 'hipDeviceAttributeWarpSize',
    88: 'hipDeviceAttributeMemoryPoolsSupported',
    89: 'hipDeviceAttributeVirtualMemoryManagementSupported',
    90: 'hipDeviceAttributeHostRegisterSupported',
    9999: 'hipDeviceAttributeCudaCompatibleEnd',
    10000: 'hipDeviceAttributeAmdSpecificBegin',
    10000: 'hipDeviceAttributeClockInstructionRate',
    10001: 'hipDeviceAttributeUnused3',
    10002: 'hipDeviceAttributeMaxSharedMemoryPerMultiprocessor',
    10003: 'hipDeviceAttributeUnused4',
    10004: 'hipDeviceAttributeUnused5',
    10005: 'hipDeviceAttributeHdpMemFlushCntl',
    10006: 'hipDeviceAttributeHdpRegFlushCntl',
    10007: 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc',
    10008: 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim',
    10009: 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim',
    10010: 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem',
    10011: 'hipDeviceAttributeIsLargeBar',
    10012: 'hipDeviceAttributeAsicRevision',
    10013: 'hipDeviceAttributeCanUseStreamWaitValue',
    10014: 'hipDeviceAttributeImageSupport',
    10015: 'hipDeviceAttributePhysicalMultiProcessorCount',
    10016: 'hipDeviceAttributeFineGrainSupport',
    10017: 'hipDeviceAttributeWallClockRate',
    19999: 'hipDeviceAttributeAmdSpecificEnd',
    20000: 'hipDeviceAttributeVendorSpecificBegin',
}
hipDeviceAttributeCudaCompatibleBegin = 0
hipDeviceAttributeEccEnabled = 0
hipDeviceAttributeAccessPolicyMaxWindowSize = 1
hipDeviceAttributeAsyncEngineCount = 2
hipDeviceAttributeCanMapHostMemory = 3
hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4
hipDeviceAttributeClockRate = 5
hipDeviceAttributeComputeMode = 6
hipDeviceAttributeComputePreemptionSupported = 7
hipDeviceAttributeConcurrentKernels = 8
hipDeviceAttributeConcurrentManagedAccess = 9
hipDeviceAttributeCooperativeLaunch = 10
hipDeviceAttributeCooperativeMultiDeviceLaunch = 11
hipDeviceAttributeDeviceOverlap = 12
hipDeviceAttributeDirectManagedMemAccessFromHost = 13
hipDeviceAttributeGlobalL1CacheSupported = 14
hipDeviceAttributeHostNativeAtomicSupported = 15
hipDeviceAttributeIntegrated = 16
hipDeviceAttributeIsMultiGpuBoard = 17
hipDeviceAttributeKernelExecTimeout = 18
hipDeviceAttributeL2CacheSize = 19
hipDeviceAttributeLocalL1CacheSupported = 20
hipDeviceAttributeLuid = 21
hipDeviceAttributeLuidDeviceNodeMask = 22
hipDeviceAttributeComputeCapabilityMajor = 23
hipDeviceAttributeManagedMemory = 24
hipDeviceAttributeMaxBlocksPerMultiProcessor = 25
hipDeviceAttributeMaxBlockDimX = 26
hipDeviceAttributeMaxBlockDimY = 27
hipDeviceAttributeMaxBlockDimZ = 28
hipDeviceAttributeMaxGridDimX = 29
hipDeviceAttributeMaxGridDimY = 30
hipDeviceAttributeMaxGridDimZ = 31
hipDeviceAttributeMaxSurface1D = 32
hipDeviceAttributeMaxSurface1DLayered = 33
hipDeviceAttributeMaxSurface2D = 34
hipDeviceAttributeMaxSurface2DLayered = 35
hipDeviceAttributeMaxSurface3D = 36
hipDeviceAttributeMaxSurfaceCubemap = 37
hipDeviceAttributeMaxSurfaceCubemapLayered = 38
hipDeviceAttributeMaxTexture1DWidth = 39
hipDeviceAttributeMaxTexture1DLayered = 40
hipDeviceAttributeMaxTexture1DLinear = 41
hipDeviceAttributeMaxTexture1DMipmap = 42
hipDeviceAttributeMaxTexture2DWidth = 43
hipDeviceAttributeMaxTexture2DHeight = 44
hipDeviceAttributeMaxTexture2DGather = 45
hipDeviceAttributeMaxTexture2DLayered = 46
hipDeviceAttributeMaxTexture2DLinear = 47
hipDeviceAttributeMaxTexture2DMipmap = 48
hipDeviceAttributeMaxTexture3DWidth = 49
hipDeviceAttributeMaxTexture3DHeight = 50
hipDeviceAttributeMaxTexture3DDepth = 51
hipDeviceAttributeMaxTexture3DAlt = 52
hipDeviceAttributeMaxTextureCubemap = 53
hipDeviceAttributeMaxTextureCubemapLayered = 54
hipDeviceAttributeMaxThreadsDim = 55
hipDeviceAttributeMaxThreadsPerBlock = 56
hipDeviceAttributeMaxThreadsPerMultiProcessor = 57
hipDeviceAttributeMaxPitch = 58
hipDeviceAttributeMemoryBusWidth = 59
hipDeviceAttributeMemoryClockRate = 60
hipDeviceAttributeComputeCapabilityMinor = 61
hipDeviceAttributeMultiGpuBoardGroupID = 62
hipDeviceAttributeMultiprocessorCount = 63
hipDeviceAttributeUnused1 = 64
hipDeviceAttributePageableMemoryAccess = 65
hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66
hipDeviceAttributePciBusId = 67
hipDeviceAttributePciDeviceId = 68
hipDeviceAttributePciDomainID = 69
hipDeviceAttributePersistingL2CacheMaxSize = 70
hipDeviceAttributeMaxRegistersPerBlock = 71
hipDeviceAttributeMaxRegistersPerMultiprocessor = 72
hipDeviceAttributeReservedSharedMemPerBlock = 73
hipDeviceAttributeMaxSharedMemoryPerBlock = 74
hipDeviceAttributeSharedMemPerBlockOptin = 75
hipDeviceAttributeSharedMemPerMultiprocessor = 76
hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77
hipDeviceAttributeStreamPrioritiesSupported = 78
hipDeviceAttributeSurfaceAlignment = 79
hipDeviceAttributeTccDriver = 80
hipDeviceAttributeTextureAlignment = 81
hipDeviceAttributeTexturePitchAlignment = 82
hipDeviceAttributeTotalConstantMemory = 83
hipDeviceAttributeTotalGlobalMem = 84
hipDeviceAttributeUnifiedAddressing = 85
hipDeviceAttributeUnused2 = 86
hipDeviceAttributeWarpSize = 87
hipDeviceAttributeMemoryPoolsSupported = 88
hipDeviceAttributeVirtualMemoryManagementSupported = 89
hipDeviceAttributeHostRegisterSupported = 90
hipDeviceAttributeCudaCompatibleEnd = 9999
hipDeviceAttributeAmdSpecificBegin = 10000
hipDeviceAttributeClockInstructionRate = 10000
hipDeviceAttributeUnused3 = 10001
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002
hipDeviceAttributeUnused4 = 10003
hipDeviceAttributeUnused5 = 10004
hipDeviceAttributeHdpMemFlushCntl = 10005
hipDeviceAttributeHdpRegFlushCntl = 10006
hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007
hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008
hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009
hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010
hipDeviceAttributeIsLargeBar = 10011
hipDeviceAttributeAsicRevision = 10012
hipDeviceAttributeCanUseStreamWaitValue = 10013
hipDeviceAttributeImageSupport = 10014
hipDeviceAttributePhysicalMultiProcessorCount = 10015
hipDeviceAttributeFineGrainSupport = 10016
hipDeviceAttributeWallClockRate = 10017
hipDeviceAttributeAmdSpecificEnd = 19999
hipDeviceAttributeVendorSpecificBegin = 20000
hipDeviceAttribute_t = ctypes.c_uint32 # enum

# values for enumeration 'hipComputeMode'
hipComputeMode__enumvalues = {
    0: 'hipComputeModeDefault',
    1: 'hipComputeModeExclusive',
    2: 'hipComputeModeProhibited',
    3: 'hipComputeModeExclusiveProcess',
}
hipComputeModeDefault = 0
hipComputeModeExclusive = 1
hipComputeModeProhibited = 2
hipComputeModeExclusiveProcess = 3
hipComputeMode = ctypes.c_uint32 # enum
hipDeviceptr_t = ctypes.POINTER(None)

# values for enumeration 'hipChannelFormatKind'
hipChannelFormatKind__enumvalues = {
    0: 'hipChannelFormatKindSigned',
    1: 'hipChannelFormatKindUnsigned',
    2: 'hipChannelFormatKindFloat',
    3: 'hipChannelFormatKindNone',
}
hipChannelFormatKindSigned = 0
hipChannelFormatKindUnsigned = 1
hipChannelFormatKindFloat = 2
hipChannelFormatKindNone = 3
hipChannelFormatKind = ctypes.c_uint32 # enum
class struct_hipChannelFormatDesc(Structure):
    pass

struct_hipChannelFormatDesc._pack_ = 1 # source:False
struct_hipChannelFormatDesc._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('z', ctypes.c_int32),
    ('w', ctypes.c_int32),
    ('f', hipChannelFormatKind),
]

hipChannelFormatDesc = struct_hipChannelFormatDesc
class struct_hipArray(Structure):
    pass

hipArray_t = ctypes.POINTER(struct_hipArray)
hipArray_const_t = ctypes.POINTER(struct_hipArray)

# values for enumeration 'hipArray_Format'
hipArray_Format__enumvalues = {
    1: 'HIP_AD_FORMAT_UNSIGNED_INT8',
    2: 'HIP_AD_FORMAT_UNSIGNED_INT16',
    3: 'HIP_AD_FORMAT_UNSIGNED_INT32',
    8: 'HIP_AD_FORMAT_SIGNED_INT8',
    9: 'HIP_AD_FORMAT_SIGNED_INT16',
    10: 'HIP_AD_FORMAT_SIGNED_INT32',
    16: 'HIP_AD_FORMAT_HALF',
    32: 'HIP_AD_FORMAT_FLOAT',
}
HIP_AD_FORMAT_UNSIGNED_INT8 = 1
HIP_AD_FORMAT_UNSIGNED_INT16 = 2
HIP_AD_FORMAT_UNSIGNED_INT32 = 3
HIP_AD_FORMAT_SIGNED_INT8 = 8
HIP_AD_FORMAT_SIGNED_INT16 = 9
HIP_AD_FORMAT_SIGNED_INT32 = 10
HIP_AD_FORMAT_HALF = 16
HIP_AD_FORMAT_FLOAT = 32
hipArray_Format = ctypes.c_uint32 # enum
class struct_HIP_ARRAY_DESCRIPTOR(Structure):
    pass

struct_HIP_ARRAY_DESCRIPTOR._pack_ = 1 # source:False
struct_HIP_ARRAY_DESCRIPTOR._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Format', hipArray_Format),
    ('NumChannels', ctypes.c_uint32),
]

HIP_ARRAY_DESCRIPTOR = struct_HIP_ARRAY_DESCRIPTOR
class struct_HIP_ARRAY3D_DESCRIPTOR(Structure):
    pass

struct_HIP_ARRAY3D_DESCRIPTOR._pack_ = 1 # source:False
struct_HIP_ARRAY3D_DESCRIPTOR._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
    ('Format', hipArray_Format),
    ('NumChannels', ctypes.c_uint32),
    ('Flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

HIP_ARRAY3D_DESCRIPTOR = struct_HIP_ARRAY3D_DESCRIPTOR
class struct_hip_Memcpy2D(Structure):
    pass

struct_hip_Memcpy2D._pack_ = 1 # source:False
struct_hip_Memcpy2D._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcMemoryType', hipMemoryType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.POINTER(None)),
    ('srcArray', ctypes.POINTER(struct_hipArray)),
    ('srcPitch', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstMemoryType', hipMemoryType),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.POINTER(None)),
    ('dstArray', ctypes.POINTER(struct_hipArray)),
    ('dstPitch', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
]

hip_Memcpy2D = struct_hip_Memcpy2D
class struct_hipMipmappedArray(Structure):
    pass

struct_hipMipmappedArray._pack_ = 1 # source:False
struct_hipMipmappedArray._fields_ = [
    ('data', ctypes.POINTER(None)),
    ('desc', struct_hipChannelFormatDesc),
    ('type', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
    ('min_mipmap_level', ctypes.c_uint32),
    ('max_mipmap_level', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('format', hipArray_Format),
    ('num_channels', ctypes.c_uint32),
]

hipMipmappedArray = struct_hipMipmappedArray
hipMipmappedArray_t = ctypes.POINTER(struct_hipMipmappedArray)
hipmipmappedArray = ctypes.POINTER(struct_hipMipmappedArray)
hipMipmappedArray_const_t = ctypes.POINTER(struct_hipMipmappedArray)

# values for enumeration 'hipResourceType'
hipResourceType__enumvalues = {
    0: 'hipResourceTypeArray',
    1: 'hipResourceTypeMipmappedArray',
    2: 'hipResourceTypeLinear',
    3: 'hipResourceTypePitch2D',
}
hipResourceTypeArray = 0
hipResourceTypeMipmappedArray = 1
hipResourceTypeLinear = 2
hipResourceTypePitch2D = 3
hipResourceType = ctypes.c_uint32 # enum

# values for enumeration 'HIPresourcetype_enum'
HIPresourcetype_enum__enumvalues = {
    0: 'HIP_RESOURCE_TYPE_ARRAY',
    1: 'HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY',
    2: 'HIP_RESOURCE_TYPE_LINEAR',
    3: 'HIP_RESOURCE_TYPE_PITCH2D',
}
HIP_RESOURCE_TYPE_ARRAY = 0
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
HIP_RESOURCE_TYPE_LINEAR = 2
HIP_RESOURCE_TYPE_PITCH2D = 3
HIPresourcetype_enum = ctypes.c_uint32 # enum
HIPresourcetype = HIPresourcetype_enum
HIPresourcetype__enumvalues = HIPresourcetype_enum__enumvalues
hipResourcetype = HIPresourcetype_enum
hipResourcetype__enumvalues = HIPresourcetype_enum__enumvalues

# values for enumeration 'HIPaddress_mode_enum'
HIPaddress_mode_enum__enumvalues = {
    0: 'HIP_TR_ADDRESS_MODE_WRAP',
    1: 'HIP_TR_ADDRESS_MODE_CLAMP',
    2: 'HIP_TR_ADDRESS_MODE_MIRROR',
    3: 'HIP_TR_ADDRESS_MODE_BORDER',
}
HIP_TR_ADDRESS_MODE_WRAP = 0
HIP_TR_ADDRESS_MODE_CLAMP = 1
HIP_TR_ADDRESS_MODE_MIRROR = 2
HIP_TR_ADDRESS_MODE_BORDER = 3
HIPaddress_mode_enum = ctypes.c_uint32 # enum
HIPaddress_mode = HIPaddress_mode_enum
HIPaddress_mode__enumvalues = HIPaddress_mode_enum__enumvalues

# values for enumeration 'HIPfilter_mode_enum'
HIPfilter_mode_enum__enumvalues = {
    0: 'HIP_TR_FILTER_MODE_POINT',
    1: 'HIP_TR_FILTER_MODE_LINEAR',
}
HIP_TR_FILTER_MODE_POINT = 0
HIP_TR_FILTER_MODE_LINEAR = 1
HIPfilter_mode_enum = ctypes.c_uint32 # enum
HIPfilter_mode = HIPfilter_mode_enum
HIPfilter_mode__enumvalues = HIPfilter_mode_enum__enumvalues
class struct_HIP_TEXTURE_DESC_st(Structure):
    pass

struct_HIP_TEXTURE_DESC_st._pack_ = 1 # source:False
struct_HIP_TEXTURE_DESC_st._fields_ = [
    ('addressMode', HIPaddress_mode_enum * 3),
    ('filterMode', HIPfilter_mode),
    ('flags', ctypes.c_uint32),
    ('maxAnisotropy', ctypes.c_uint32),
    ('mipmapFilterMode', HIPfilter_mode),
    ('mipmapLevelBias', ctypes.c_float),
    ('minMipmapLevelClamp', ctypes.c_float),
    ('maxMipmapLevelClamp', ctypes.c_float),
    ('borderColor', ctypes.c_float * 4),
    ('reserved', ctypes.c_int32 * 12),
]

HIP_TEXTURE_DESC = struct_HIP_TEXTURE_DESC_st

# values for enumeration 'hipResourceViewFormat'
hipResourceViewFormat__enumvalues = {
    0: 'hipResViewFormatNone',
    1: 'hipResViewFormatUnsignedChar1',
    2: 'hipResViewFormatUnsignedChar2',
    3: 'hipResViewFormatUnsignedChar4',
    4: 'hipResViewFormatSignedChar1',
    5: 'hipResViewFormatSignedChar2',
    6: 'hipResViewFormatSignedChar4',
    7: 'hipResViewFormatUnsignedShort1',
    8: 'hipResViewFormatUnsignedShort2',
    9: 'hipResViewFormatUnsignedShort4',
    10: 'hipResViewFormatSignedShort1',
    11: 'hipResViewFormatSignedShort2',
    12: 'hipResViewFormatSignedShort4',
    13: 'hipResViewFormatUnsignedInt1',
    14: 'hipResViewFormatUnsignedInt2',
    15: 'hipResViewFormatUnsignedInt4',
    16: 'hipResViewFormatSignedInt1',
    17: 'hipResViewFormatSignedInt2',
    18: 'hipResViewFormatSignedInt4',
    19: 'hipResViewFormatHalf1',
    20: 'hipResViewFormatHalf2',
    21: 'hipResViewFormatHalf4',
    22: 'hipResViewFormatFloat1',
    23: 'hipResViewFormatFloat2',
    24: 'hipResViewFormatFloat4',
    25: 'hipResViewFormatUnsignedBlockCompressed1',
    26: 'hipResViewFormatUnsignedBlockCompressed2',
    27: 'hipResViewFormatUnsignedBlockCompressed3',
    28: 'hipResViewFormatUnsignedBlockCompressed4',
    29: 'hipResViewFormatSignedBlockCompressed4',
    30: 'hipResViewFormatUnsignedBlockCompressed5',
    31: 'hipResViewFormatSignedBlockCompressed5',
    32: 'hipResViewFormatUnsignedBlockCompressed6H',
    33: 'hipResViewFormatSignedBlockCompressed6H',
    34: 'hipResViewFormatUnsignedBlockCompressed7',
}
hipResViewFormatNone = 0
hipResViewFormatUnsignedChar1 = 1
hipResViewFormatUnsignedChar2 = 2
hipResViewFormatUnsignedChar4 = 3
hipResViewFormatSignedChar1 = 4
hipResViewFormatSignedChar2 = 5
hipResViewFormatSignedChar4 = 6
hipResViewFormatUnsignedShort1 = 7
hipResViewFormatUnsignedShort2 = 8
hipResViewFormatUnsignedShort4 = 9
hipResViewFormatSignedShort1 = 10
hipResViewFormatSignedShort2 = 11
hipResViewFormatSignedShort4 = 12
hipResViewFormatUnsignedInt1 = 13
hipResViewFormatUnsignedInt2 = 14
hipResViewFormatUnsignedInt4 = 15
hipResViewFormatSignedInt1 = 16
hipResViewFormatSignedInt2 = 17
hipResViewFormatSignedInt4 = 18
hipResViewFormatHalf1 = 19
hipResViewFormatHalf2 = 20
hipResViewFormatHalf4 = 21
hipResViewFormatFloat1 = 22
hipResViewFormatFloat2 = 23
hipResViewFormatFloat4 = 24
hipResViewFormatUnsignedBlockCompressed1 = 25
hipResViewFormatUnsignedBlockCompressed2 = 26
hipResViewFormatUnsignedBlockCompressed3 = 27
hipResViewFormatUnsignedBlockCompressed4 = 28
hipResViewFormatSignedBlockCompressed4 = 29
hipResViewFormatUnsignedBlockCompressed5 = 30
hipResViewFormatSignedBlockCompressed5 = 31
hipResViewFormatUnsignedBlockCompressed6H = 32
hipResViewFormatSignedBlockCompressed6H = 33
hipResViewFormatUnsignedBlockCompressed7 = 34
hipResourceViewFormat = ctypes.c_uint32 # enum

# values for enumeration 'HIPresourceViewFormat_enum'
HIPresourceViewFormat_enum__enumvalues = {
    0: 'HIP_RES_VIEW_FORMAT_NONE',
    1: 'HIP_RES_VIEW_FORMAT_UINT_1X8',
    2: 'HIP_RES_VIEW_FORMAT_UINT_2X8',
    3: 'HIP_RES_VIEW_FORMAT_UINT_4X8',
    4: 'HIP_RES_VIEW_FORMAT_SINT_1X8',
    5: 'HIP_RES_VIEW_FORMAT_SINT_2X8',
    6: 'HIP_RES_VIEW_FORMAT_SINT_4X8',
    7: 'HIP_RES_VIEW_FORMAT_UINT_1X16',
    8: 'HIP_RES_VIEW_FORMAT_UINT_2X16',
    9: 'HIP_RES_VIEW_FORMAT_UINT_4X16',
    10: 'HIP_RES_VIEW_FORMAT_SINT_1X16',
    11: 'HIP_RES_VIEW_FORMAT_SINT_2X16',
    12: 'HIP_RES_VIEW_FORMAT_SINT_4X16',
    13: 'HIP_RES_VIEW_FORMAT_UINT_1X32',
    14: 'HIP_RES_VIEW_FORMAT_UINT_2X32',
    15: 'HIP_RES_VIEW_FORMAT_UINT_4X32',
    16: 'HIP_RES_VIEW_FORMAT_SINT_1X32',
    17: 'HIP_RES_VIEW_FORMAT_SINT_2X32',
    18: 'HIP_RES_VIEW_FORMAT_SINT_4X32',
    19: 'HIP_RES_VIEW_FORMAT_FLOAT_1X16',
    20: 'HIP_RES_VIEW_FORMAT_FLOAT_2X16',
    21: 'HIP_RES_VIEW_FORMAT_FLOAT_4X16',
    22: 'HIP_RES_VIEW_FORMAT_FLOAT_1X32',
    23: 'HIP_RES_VIEW_FORMAT_FLOAT_2X32',
    24: 'HIP_RES_VIEW_FORMAT_FLOAT_4X32',
    25: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC1',
    26: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC2',
    27: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC3',
    28: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC4',
    29: 'HIP_RES_VIEW_FORMAT_SIGNED_BC4',
    30: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC5',
    31: 'HIP_RES_VIEW_FORMAT_SIGNED_BC5',
    32: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    33: 'HIP_RES_VIEW_FORMAT_SIGNED_BC6H',
    34: 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC7',
}
HIP_RES_VIEW_FORMAT_NONE = 0
HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
HIPresourceViewFormat_enum = ctypes.c_uint32 # enum
HIPresourceViewFormat = HIPresourceViewFormat_enum
HIPresourceViewFormat__enumvalues = HIPresourceViewFormat_enum__enumvalues
class struct_hipResourceDesc(Structure):
    pass

class union_hipResourceDesc_res(Union):
    pass

class struct_hipResourceDesc_0_array(Structure):
    pass

struct_hipResourceDesc_0_array._pack_ = 1 # source:False
struct_hipResourceDesc_0_array._fields_ = [
    ('array', ctypes.POINTER(struct_hipArray)),
]

class struct_hipResourceDesc_0_mipmap(Structure):
    pass

struct_hipResourceDesc_0_mipmap._pack_ = 1 # source:False
struct_hipResourceDesc_0_mipmap._fields_ = [
    ('mipmap', ctypes.POINTER(struct_hipMipmappedArray)),
]

class struct_hipResourceDesc_0_linear(Structure):
    pass

struct_hipResourceDesc_0_linear._pack_ = 1 # source:False
struct_hipResourceDesc_0_linear._fields_ = [
    ('devPtr', ctypes.POINTER(None)),
    ('desc', struct_hipChannelFormatDesc),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('sizeInBytes', ctypes.c_uint64),
]

class struct_hipResourceDesc_0_pitch2D(Structure):
    pass

struct_hipResourceDesc_0_pitch2D._pack_ = 1 # source:False
struct_hipResourceDesc_0_pitch2D._fields_ = [
    ('devPtr', ctypes.POINTER(None)),
    ('desc', struct_hipChannelFormatDesc),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('pitchInBytes', ctypes.c_uint64),
]

union_hipResourceDesc_res._pack_ = 1 # source:False
union_hipResourceDesc_res._fields_ = [
    ('array', struct_hipResourceDesc_0_array),
    ('mipmap', struct_hipResourceDesc_0_mipmap),
    ('linear', struct_hipResourceDesc_0_linear),
    ('pitch2D', struct_hipResourceDesc_0_pitch2D),
]

struct_hipResourceDesc._pack_ = 1 # source:False
struct_hipResourceDesc._fields_ = [
    ('resType', hipResourceType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('res', union_hipResourceDesc_res),
]

hipResourceDesc = struct_hipResourceDesc
class struct_HIP_RESOURCE_DESC_st(Structure):
    pass

class union_HIP_RESOURCE_DESC_st_res(Union):
    pass

class struct_HIP_RESOURCE_DESC_st_0_array(Structure):
    pass

struct_HIP_RESOURCE_DESC_st_0_array._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st_0_array._fields_ = [
    ('hArray', ctypes.POINTER(struct_hipArray)),
]

class struct_HIP_RESOURCE_DESC_st_0_mipmap(Structure):
    pass

struct_HIP_RESOURCE_DESC_st_0_mipmap._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st_0_mipmap._fields_ = [
    ('hMipmappedArray', ctypes.POINTER(struct_hipMipmappedArray)),
]

class struct_HIP_RESOURCE_DESC_st_0_linear(Structure):
    pass

struct_HIP_RESOURCE_DESC_st_0_linear._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st_0_linear._fields_ = [
    ('devPtr', ctypes.POINTER(None)),
    ('format', hipArray_Format),
    ('numChannels', ctypes.c_uint32),
    ('sizeInBytes', ctypes.c_uint64),
]

class struct_HIP_RESOURCE_DESC_st_0_pitch2D(Structure):
    pass

struct_HIP_RESOURCE_DESC_st_0_pitch2D._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st_0_pitch2D._fields_ = [
    ('devPtr', ctypes.POINTER(None)),
    ('format', hipArray_Format),
    ('numChannels', ctypes.c_uint32),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('pitchInBytes', ctypes.c_uint64),
]

class struct_HIP_RESOURCE_DESC_st_0_reserved(Structure):
    pass

struct_HIP_RESOURCE_DESC_st_0_reserved._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st_0_reserved._fields_ = [
    ('reserved', ctypes.c_int32 * 32),
]

union_HIP_RESOURCE_DESC_st_res._pack_ = 1 # source:False
union_HIP_RESOURCE_DESC_st_res._fields_ = [
    ('array', struct_HIP_RESOURCE_DESC_st_0_array),
    ('mipmap', struct_HIP_RESOURCE_DESC_st_0_mipmap),
    ('linear', struct_HIP_RESOURCE_DESC_st_0_linear),
    ('pitch2D', struct_HIP_RESOURCE_DESC_st_0_pitch2D),
    ('reserved', struct_HIP_RESOURCE_DESC_st_0_reserved),
]

struct_HIP_RESOURCE_DESC_st._pack_ = 1 # source:False
struct_HIP_RESOURCE_DESC_st._fields_ = [
    ('resType', HIPresourcetype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('res', union_HIP_RESOURCE_DESC_st_res),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

HIP_RESOURCE_DESC = struct_HIP_RESOURCE_DESC_st
class struct_hipResourceViewDesc(Structure):
    pass

struct_hipResourceViewDesc._pack_ = 1 # source:False
struct_hipResourceViewDesc._fields_ = [
    ('format', hipResourceViewFormat),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('firstMipmapLevel', ctypes.c_uint32),
    ('lastMipmapLevel', ctypes.c_uint32),
    ('firstLayer', ctypes.c_uint32),
    ('lastLayer', ctypes.c_uint32),
]

class struct_HIP_RESOURCE_VIEW_DESC_st(Structure):
    pass

struct_HIP_RESOURCE_VIEW_DESC_st._pack_ = 1 # source:False
struct_HIP_RESOURCE_VIEW_DESC_st._fields_ = [
    ('format', HIPresourceViewFormat),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('firstMipmapLevel', ctypes.c_uint32),
    ('lastMipmapLevel', ctypes.c_uint32),
    ('firstLayer', ctypes.c_uint32),
    ('lastLayer', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
]

HIP_RESOURCE_VIEW_DESC = struct_HIP_RESOURCE_VIEW_DESC_st

# values for enumeration 'hipMemcpyKind'
hipMemcpyKind__enumvalues = {
    0: 'hipMemcpyHostToHost',
    1: 'hipMemcpyHostToDevice',
    2: 'hipMemcpyDeviceToHost',
    3: 'hipMemcpyDeviceToDevice',
    4: 'hipMemcpyDefault',
}
hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4
hipMemcpyKind = ctypes.c_uint32 # enum
class struct_hipPitchedPtr(Structure):
    pass

struct_hipPitchedPtr._pack_ = 1 # source:False
struct_hipPitchedPtr._fields_ = [
    ('ptr', ctypes.POINTER(None)),
    ('pitch', ctypes.c_uint64),
    ('xsize', ctypes.c_uint64),
    ('ysize', ctypes.c_uint64),
]

hipPitchedPtr = struct_hipPitchedPtr
class struct_hipExtent(Structure):
    pass

struct_hipExtent._pack_ = 1 # source:False
struct_hipExtent._fields_ = [
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
]

hipExtent = struct_hipExtent
class struct_hipPos(Structure):
    pass

struct_hipPos._pack_ = 1 # source:False
struct_hipPos._fields_ = [
    ('x', ctypes.c_uint64),
    ('y', ctypes.c_uint64),
    ('z', ctypes.c_uint64),
]

hipPos = struct_hipPos
class struct_hipMemcpy3DParms(Structure):
    pass

struct_hipMemcpy3DParms._pack_ = 1 # source:False
struct_hipMemcpy3DParms._fields_ = [
    ('srcArray', ctypes.POINTER(struct_hipArray)),
    ('srcPos', struct_hipPos),
    ('srcPtr', struct_hipPitchedPtr),
    ('dstArray', ctypes.POINTER(struct_hipArray)),
    ('dstPos', struct_hipPos),
    ('dstPtr', struct_hipPitchedPtr),
    ('extent', struct_hipExtent),
    ('kind', hipMemcpyKind),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipMemcpy3DParms = struct_hipMemcpy3DParms
class struct_HIP_MEMCPY3D(Structure):
    pass

struct_HIP_MEMCPY3D._pack_ = 1 # source:False
struct_HIP_MEMCPY3D._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcZ', ctypes.c_uint64),
    ('srcLOD', ctypes.c_uint64),
    ('srcMemoryType', hipMemoryType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.POINTER(None)),
    ('srcArray', ctypes.POINTER(struct_hipArray)),
    ('srcPitch', ctypes.c_uint64),
    ('srcHeight', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstZ', ctypes.c_uint64),
    ('dstLOD', ctypes.c_uint64),
    ('dstMemoryType', hipMemoryType),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.POINTER(None)),
    ('dstArray', ctypes.POINTER(struct_hipArray)),
    ('dstPitch', ctypes.c_uint64),
    ('dstHeight', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
]

HIP_MEMCPY3D = struct_HIP_MEMCPY3D
size_t = ctypes.c_uint64
try:
    make_hipPitchedPtr = _libraries['FIXME_STUB'].make_hipPitchedPtr
    make_hipPitchedPtr.restype = struct_hipPitchedPtr
    make_hipPitchedPtr.argtypes = [ctypes.POINTER(None), size_t, size_t, size_t]
except AttributeError:
    pass
try:
    make_hipPos = _libraries['FIXME_STUB'].make_hipPos
    make_hipPos.restype = struct_hipPos
    make_hipPos.argtypes = [size_t, size_t, size_t]
except AttributeError:
    pass
try:
    make_hipExtent = _libraries['FIXME_STUB'].make_hipExtent
    make_hipExtent.restype = struct_hipExtent
    make_hipExtent.argtypes = [size_t, size_t, size_t]
except AttributeError:
    pass

# values for enumeration 'hipFunction_attribute'
hipFunction_attribute__enumvalues = {
    0: 'HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    1: 'HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    2: 'HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    3: 'HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
    4: 'HIP_FUNC_ATTRIBUTE_NUM_REGS',
    5: 'HIP_FUNC_ATTRIBUTE_PTX_VERSION',
    6: 'HIP_FUNC_ATTRIBUTE_BINARY_VERSION',
    7: 'HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    8: 'HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    9: 'HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    10: 'HIP_FUNC_ATTRIBUTE_MAX',
}
HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5
HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6
HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
HIP_FUNC_ATTRIBUTE_MAX = 10
hipFunction_attribute = ctypes.c_uint32 # enum

# values for enumeration 'hipPointer_attribute'
hipPointer_attribute__enumvalues = {
    1: 'HIP_POINTER_ATTRIBUTE_CONTEXT',
    2: 'HIP_POINTER_ATTRIBUTE_MEMORY_TYPE',
    3: 'HIP_POINTER_ATTRIBUTE_DEVICE_POINTER',
    4: 'HIP_POINTER_ATTRIBUTE_HOST_POINTER',
    5: 'HIP_POINTER_ATTRIBUTE_P2P_TOKENS',
    6: 'HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS',
    7: 'HIP_POINTER_ATTRIBUTE_BUFFER_ID',
    8: 'HIP_POINTER_ATTRIBUTE_IS_MANAGED',
    9: 'HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    10: 'HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE',
    11: 'HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    12: 'HIP_POINTER_ATTRIBUTE_RANGE_SIZE',
    13: 'HIP_POINTER_ATTRIBUTE_MAPPED',
    14: 'HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    15: 'HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    16: 'HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    17: 'HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
}
HIP_POINTER_ATTRIBUTE_CONTEXT = 1
HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4
HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5
HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7
HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8
HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10
HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12
HIP_POINTER_ATTRIBUTE_MAPPED = 13
HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
hipPointer_attribute = ctypes.c_uint32 # enum
try:
    hip_init = _libraries['FIXME_STUB'].hip_init
    hip_init.restype = hipError_t
    hip_init.argtypes = []
except AttributeError:
    pass
class struct_ihipCtx_t(Structure):
    pass

hipCtx_t = ctypes.POINTER(struct_ihipCtx_t)
hipDevice_t = ctypes.c_int32

# values for enumeration 'hipDeviceP2PAttr'
hipDeviceP2PAttr__enumvalues = {
    0: 'hipDevP2PAttrPerformanceRank',
    1: 'hipDevP2PAttrAccessSupported',
    2: 'hipDevP2PAttrNativeAtomicSupported',
    3: 'hipDevP2PAttrHipArrayAccessSupported',
}
hipDevP2PAttrPerformanceRank = 0
hipDevP2PAttrAccessSupported = 1
hipDevP2PAttrNativeAtomicSupported = 2
hipDevP2PAttrHipArrayAccessSupported = 3
hipDeviceP2PAttr = ctypes.c_uint32 # enum
class struct_ihipStream_t(Structure):
    pass

hipStream_t = ctypes.POINTER(struct_ihipStream_t)
class struct_hipIpcMemHandle_st(Structure):
    pass

struct_hipIpcMemHandle_st._pack_ = 1 # source:False
struct_hipIpcMemHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

hipIpcMemHandle_t = struct_hipIpcMemHandle_st
class struct_hipIpcEventHandle_st(Structure):
    pass

struct_hipIpcEventHandle_st._pack_ = 1 # source:False
struct_hipIpcEventHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

hipIpcEventHandle_t = struct_hipIpcEventHandle_st
class struct_ihipModule_t(Structure):
    pass

hipModule_t = ctypes.POINTER(struct_ihipModule_t)
class struct_ihipModuleSymbol_t(Structure):
    pass

hipFunction_t = ctypes.POINTER(struct_ihipModuleSymbol_t)
class struct_ihipMemPoolHandle_t(Structure):
    pass

hipMemPool_t = ctypes.POINTER(struct_ihipMemPoolHandle_t)
class struct_hipFuncAttributes(Structure):
    pass

struct_hipFuncAttributes._pack_ = 1 # source:False
struct_hipFuncAttributes._fields_ = [
    ('binaryVersion', ctypes.c_int32),
    ('cacheModeCA', ctypes.c_int32),
    ('constSizeBytes', ctypes.c_uint64),
    ('localSizeBytes', ctypes.c_uint64),
    ('maxDynamicSharedSizeBytes', ctypes.c_int32),
    ('maxThreadsPerBlock', ctypes.c_int32),
    ('numRegs', ctypes.c_int32),
    ('preferredShmemCarveout', ctypes.c_int32),
    ('ptxVersion', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('sharedSizeBytes', ctypes.c_uint64),
]

hipFuncAttributes = struct_hipFuncAttributes
class struct_ihipEvent_t(Structure):
    pass

hipEvent_t = ctypes.POINTER(struct_ihipEvent_t)

# values for enumeration 'hipLimit_t'
hipLimit_t__enumvalues = {
    0: 'hipLimitStackSize',
    1: 'hipLimitPrintfFifoSize',
    2: 'hipLimitMallocHeapSize',
    3: 'hipLimitRange',
}
hipLimitStackSize = 0
hipLimitPrintfFifoSize = 1
hipLimitMallocHeapSize = 2
hipLimitRange = 3
hipLimit_t = ctypes.c_uint32 # enum

# values for enumeration 'hipMemoryAdvise'
hipMemoryAdvise__enumvalues = {
    1: 'hipMemAdviseSetReadMostly',
    2: 'hipMemAdviseUnsetReadMostly',
    3: 'hipMemAdviseSetPreferredLocation',
    4: 'hipMemAdviseUnsetPreferredLocation',
    5: 'hipMemAdviseSetAccessedBy',
    6: 'hipMemAdviseUnsetAccessedBy',
    100: 'hipMemAdviseSetCoarseGrain',
    101: 'hipMemAdviseUnsetCoarseGrain',
}
hipMemAdviseSetReadMostly = 1
hipMemAdviseUnsetReadMostly = 2
hipMemAdviseSetPreferredLocation = 3
hipMemAdviseUnsetPreferredLocation = 4
hipMemAdviseSetAccessedBy = 5
hipMemAdviseUnsetAccessedBy = 6
hipMemAdviseSetCoarseGrain = 100
hipMemAdviseUnsetCoarseGrain = 101
hipMemoryAdvise = ctypes.c_uint32 # enum

# values for enumeration 'hipMemRangeCoherencyMode'
hipMemRangeCoherencyMode__enumvalues = {
    0: 'hipMemRangeCoherencyModeFineGrain',
    1: 'hipMemRangeCoherencyModeCoarseGrain',
    2: 'hipMemRangeCoherencyModeIndeterminate',
}
hipMemRangeCoherencyModeFineGrain = 0
hipMemRangeCoherencyModeCoarseGrain = 1
hipMemRangeCoherencyModeIndeterminate = 2
hipMemRangeCoherencyMode = ctypes.c_uint32 # enum

# values for enumeration 'hipMemRangeAttribute'
hipMemRangeAttribute__enumvalues = {
    1: 'hipMemRangeAttributeReadMostly',
    2: 'hipMemRangeAttributePreferredLocation',
    3: 'hipMemRangeAttributeAccessedBy',
    4: 'hipMemRangeAttributeLastPrefetchLocation',
    100: 'hipMemRangeAttributeCoherencyMode',
}
hipMemRangeAttributeReadMostly = 1
hipMemRangeAttributePreferredLocation = 2
hipMemRangeAttributeAccessedBy = 3
hipMemRangeAttributeLastPrefetchLocation = 4
hipMemRangeAttributeCoherencyMode = 100
hipMemRangeAttribute = ctypes.c_uint32 # enum

# values for enumeration 'hipMemPoolAttr'
hipMemPoolAttr__enumvalues = {
    1: 'hipMemPoolReuseFollowEventDependencies',
    2: 'hipMemPoolReuseAllowOpportunistic',
    3: 'hipMemPoolReuseAllowInternalDependencies',
    4: 'hipMemPoolAttrReleaseThreshold',
    5: 'hipMemPoolAttrReservedMemCurrent',
    6: 'hipMemPoolAttrReservedMemHigh',
    7: 'hipMemPoolAttrUsedMemCurrent',
    8: 'hipMemPoolAttrUsedMemHigh',
}
hipMemPoolReuseFollowEventDependencies = 1
hipMemPoolReuseAllowOpportunistic = 2
hipMemPoolReuseAllowInternalDependencies = 3
hipMemPoolAttrReleaseThreshold = 4
hipMemPoolAttrReservedMemCurrent = 5
hipMemPoolAttrReservedMemHigh = 6
hipMemPoolAttrUsedMemCurrent = 7
hipMemPoolAttrUsedMemHigh = 8
hipMemPoolAttr = ctypes.c_uint32 # enum

# values for enumeration 'hipMemLocationType'
hipMemLocationType__enumvalues = {
    0: 'hipMemLocationTypeInvalid',
    1: 'hipMemLocationTypeDevice',
}
hipMemLocationTypeInvalid = 0
hipMemLocationTypeDevice = 1
hipMemLocationType = ctypes.c_uint32 # enum
class struct_hipMemLocation(Structure):
    pass

struct_hipMemLocation._pack_ = 1 # source:False
struct_hipMemLocation._fields_ = [
    ('type', hipMemLocationType),
    ('id', ctypes.c_int32),
]

hipMemLocation = struct_hipMemLocation

# values for enumeration 'hipMemAccessFlags'
hipMemAccessFlags__enumvalues = {
    0: 'hipMemAccessFlagsProtNone',
    1: 'hipMemAccessFlagsProtRead',
    3: 'hipMemAccessFlagsProtReadWrite',
}
hipMemAccessFlagsProtNone = 0
hipMemAccessFlagsProtRead = 1
hipMemAccessFlagsProtReadWrite = 3
hipMemAccessFlags = ctypes.c_uint32 # enum
class struct_hipMemAccessDesc(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('location', hipMemLocation),
    ('flags', hipMemAccessFlags),
     ]

hipMemAccessDesc = struct_hipMemAccessDesc

# values for enumeration 'hipMemAllocationType'
hipMemAllocationType__enumvalues = {
    0: 'hipMemAllocationTypeInvalid',
    1: 'hipMemAllocationTypePinned',
    2147483647: 'hipMemAllocationTypeMax',
}
hipMemAllocationTypeInvalid = 0
hipMemAllocationTypePinned = 1
hipMemAllocationTypeMax = 2147483647
hipMemAllocationType = ctypes.c_uint32 # enum

# values for enumeration 'hipMemAllocationHandleType'
hipMemAllocationHandleType__enumvalues = {
    0: 'hipMemHandleTypeNone',
    1: 'hipMemHandleTypePosixFileDescriptor',
    2: 'hipMemHandleTypeWin32',
    4: 'hipMemHandleTypeWin32Kmt',
}
hipMemHandleTypeNone = 0
hipMemHandleTypePosixFileDescriptor = 1
hipMemHandleTypeWin32 = 2
hipMemHandleTypeWin32Kmt = 4
hipMemAllocationHandleType = ctypes.c_uint32 # enum
class struct_hipMemPoolProps(Structure):
    pass

struct_hipMemPoolProps._pack_ = 1 # source:False
struct_hipMemPoolProps._fields_ = [
    ('allocType', hipMemAllocationType),
    ('handleTypes', hipMemAllocationHandleType),
    ('location', hipMemLocation),
    ('win32SecurityAttributes', ctypes.POINTER(None)),
    ('reserved', ctypes.c_ubyte * 64),
]

hipMemPoolProps = struct_hipMemPoolProps
class struct_hipMemPoolPtrExportData(Structure):
    pass

struct_hipMemPoolPtrExportData._pack_ = 1 # source:False
struct_hipMemPoolPtrExportData._fields_ = [
    ('reserved', ctypes.c_ubyte * 64),
]

hipMemPoolPtrExportData = struct_hipMemPoolPtrExportData

# values for enumeration 'hipJitOption'
hipJitOption__enumvalues = {
    0: 'hipJitOptionMaxRegisters',
    1: 'hipJitOptionThreadsPerBlock',
    2: 'hipJitOptionWallTime',
    3: 'hipJitOptionInfoLogBuffer',
    4: 'hipJitOptionInfoLogBufferSizeBytes',
    5: 'hipJitOptionErrorLogBuffer',
    6: 'hipJitOptionErrorLogBufferSizeBytes',
    7: 'hipJitOptionOptimizationLevel',
    8: 'hipJitOptionTargetFromContext',
    9: 'hipJitOptionTarget',
    10: 'hipJitOptionFallbackStrategy',
    11: 'hipJitOptionGenerateDebugInfo',
    12: 'hipJitOptionLogVerbose',
    13: 'hipJitOptionGenerateLineInfo',
    14: 'hipJitOptionCacheMode',
    15: 'hipJitOptionSm3xOpt',
    16: 'hipJitOptionFastCompile',
    17: 'hipJitOptionNumOptions',
}
hipJitOptionMaxRegisters = 0
hipJitOptionThreadsPerBlock = 1
hipJitOptionWallTime = 2
hipJitOptionInfoLogBuffer = 3
hipJitOptionInfoLogBufferSizeBytes = 4
hipJitOptionErrorLogBuffer = 5
hipJitOptionErrorLogBufferSizeBytes = 6
hipJitOptionOptimizationLevel = 7
hipJitOptionTargetFromContext = 8
hipJitOptionTarget = 9
hipJitOptionFallbackStrategy = 10
hipJitOptionGenerateDebugInfo = 11
hipJitOptionLogVerbose = 12
hipJitOptionGenerateLineInfo = 13
hipJitOptionCacheMode = 14
hipJitOptionSm3xOpt = 15
hipJitOptionFastCompile = 16
hipJitOptionNumOptions = 17
hipJitOption = ctypes.c_uint32 # enum

# values for enumeration 'hipFuncAttribute'
hipFuncAttribute__enumvalues = {
    8: 'hipFuncAttributeMaxDynamicSharedMemorySize',
    9: 'hipFuncAttributePreferredSharedMemoryCarveout',
    10: 'hipFuncAttributeMax',
}
hipFuncAttributeMaxDynamicSharedMemorySize = 8
hipFuncAttributePreferredSharedMemoryCarveout = 9
hipFuncAttributeMax = 10
hipFuncAttribute = ctypes.c_uint32 # enum

# values for enumeration 'hipFuncCache_t'
hipFuncCache_t__enumvalues = {
    0: 'hipFuncCachePreferNone',
    1: 'hipFuncCachePreferShared',
    2: 'hipFuncCachePreferL1',
    3: 'hipFuncCachePreferEqual',
}
hipFuncCachePreferNone = 0
hipFuncCachePreferShared = 1
hipFuncCachePreferL1 = 2
hipFuncCachePreferEqual = 3
hipFuncCache_t = ctypes.c_uint32 # enum

# values for enumeration 'hipSharedMemConfig'
hipSharedMemConfig__enumvalues = {
    0: 'hipSharedMemBankSizeDefault',
    1: 'hipSharedMemBankSizeFourByte',
    2: 'hipSharedMemBankSizeEightByte',
}
hipSharedMemBankSizeDefault = 0
hipSharedMemBankSizeFourByte = 1
hipSharedMemBankSizeEightByte = 2
hipSharedMemConfig = ctypes.c_uint32 # enum
class struct_dim3(Structure):
    pass

struct_dim3._pack_ = 1 # source:False
struct_dim3._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
]

dim3 = struct_dim3
class struct_hipLaunchParams_t(Structure):
    pass

struct_hipLaunchParams_t._pack_ = 1 # source:False
struct_hipLaunchParams_t._fields_ = [
    ('func', ctypes.POINTER(None)),
    ('gridDim', dim3),
    ('blockDim', dim3),
    ('args', ctypes.POINTER(ctypes.POINTER(None))),
    ('sharedMem', ctypes.c_uint64),
    ('stream', ctypes.POINTER(struct_ihipStream_t)),
]

hipLaunchParams = struct_hipLaunchParams_t
class struct_hipFunctionLaunchParams_t(Structure):
    pass

struct_hipFunctionLaunchParams_t._pack_ = 1 # source:False
struct_hipFunctionLaunchParams_t._fields_ = [
    ('function', ctypes.POINTER(struct_ihipModuleSymbol_t)),
    ('gridDimX', ctypes.c_uint32),
    ('gridDimY', ctypes.c_uint32),
    ('gridDimZ', ctypes.c_uint32),
    ('blockDimX', ctypes.c_uint32),
    ('blockDimY', ctypes.c_uint32),
    ('blockDimZ', ctypes.c_uint32),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('hStream', ctypes.POINTER(struct_ihipStream_t)),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
]

hipFunctionLaunchParams = struct_hipFunctionLaunchParams_t

# values for enumeration 'hipExternalMemoryHandleType_enum'
hipExternalMemoryHandleType_enum__enumvalues = {
    1: 'hipExternalMemoryHandleTypeOpaqueFd',
    2: 'hipExternalMemoryHandleTypeOpaqueWin32',
    3: 'hipExternalMemoryHandleTypeOpaqueWin32Kmt',
    4: 'hipExternalMemoryHandleTypeD3D12Heap',
    5: 'hipExternalMemoryHandleTypeD3D12Resource',
    6: 'hipExternalMemoryHandleTypeD3D11Resource',
    7: 'hipExternalMemoryHandleTypeD3D11ResourceKmt',
    8: 'hipExternalMemoryHandleTypeNvSciBuf',
}
hipExternalMemoryHandleTypeOpaqueFd = 1
hipExternalMemoryHandleTypeOpaqueWin32 = 2
hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
hipExternalMemoryHandleTypeD3D12Heap = 4
hipExternalMemoryHandleTypeD3D12Resource = 5
hipExternalMemoryHandleTypeD3D11Resource = 6
hipExternalMemoryHandleTypeD3D11ResourceKmt = 7
hipExternalMemoryHandleTypeNvSciBuf = 8
hipExternalMemoryHandleType_enum = ctypes.c_uint32 # enum
hipExternalMemoryHandleType = hipExternalMemoryHandleType_enum
hipExternalMemoryHandleType__enumvalues = hipExternalMemoryHandleType_enum__enumvalues
class struct_hipExternalMemoryHandleDesc_st(Structure):
    pass

class union_hipExternalMemoryHandleDesc_st_handle(Union):
    pass

class struct_hipExternalMemoryHandleDesc_st_0_win32(Structure):
    pass

struct_hipExternalMemoryHandleDesc_st_0_win32._pack_ = 1 # source:False
struct_hipExternalMemoryHandleDesc_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_hipExternalMemoryHandleDesc_st_handle._pack_ = 1 # source:False
union_hipExternalMemoryHandleDesc_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_hipExternalMemoryHandleDesc_st_0_win32),
    ('nvSciBufObject', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_hipExternalMemoryHandleDesc_st._pack_ = 1 # source:False
struct_hipExternalMemoryHandleDesc_st._fields_ = [
    ('type', hipExternalMemoryHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_hipExternalMemoryHandleDesc_st_handle),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

hipExternalMemoryHandleDesc = struct_hipExternalMemoryHandleDesc_st
class struct_hipExternalMemoryBufferDesc_st(Structure):
    pass

struct_hipExternalMemoryBufferDesc_st._pack_ = 1 # source:False
struct_hipExternalMemoryBufferDesc_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipExternalMemoryBufferDesc = struct_hipExternalMemoryBufferDesc_st
class struct_hipExternalMemoryMipmappedArrayDesc_st(Structure):
    pass

struct_hipExternalMemoryMipmappedArrayDesc_st._pack_ = 1 # source:False
struct_hipExternalMemoryMipmappedArrayDesc_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('formatDesc', hipChannelFormatDesc),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('extent', hipExtent),
    ('flags', ctypes.c_uint32),
    ('numLevels', ctypes.c_uint32),
]

hipExternalMemoryMipmappedArrayDesc = struct_hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t = ctypes.POINTER(None)

# values for enumeration 'hipExternalSemaphoreHandleType_enum'
hipExternalSemaphoreHandleType_enum__enumvalues = {
    1: 'hipExternalSemaphoreHandleTypeOpaqueFd',
    2: 'hipExternalSemaphoreHandleTypeOpaqueWin32',
    3: 'hipExternalSemaphoreHandleTypeOpaqueWin32Kmt',
    4: 'hipExternalSemaphoreHandleTypeD3D12Fence',
    5: 'hipExternalSemaphoreHandleTypeD3D11Fence',
    6: 'hipExternalSemaphoreHandleTypeNvSciSync',
    7: 'hipExternalSemaphoreHandleTypeKeyedMutex',
    8: 'hipExternalSemaphoreHandleTypeKeyedMutexKmt',
    9: 'hipExternalSemaphoreHandleTypeTimelineSemaphoreFd',
    10: 'hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32',
}
hipExternalSemaphoreHandleTypeOpaqueFd = 1
hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
hipExternalSemaphoreHandleTypeD3D12Fence = 4
hipExternalSemaphoreHandleTypeD3D11Fence = 5
hipExternalSemaphoreHandleTypeNvSciSync = 6
hipExternalSemaphoreHandleTypeKeyedMutex = 7
hipExternalSemaphoreHandleTypeKeyedMutexKmt = 8
hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9
hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10
hipExternalSemaphoreHandleType_enum = ctypes.c_uint32 # enum
hipExternalSemaphoreHandleType = hipExternalSemaphoreHandleType_enum
hipExternalSemaphoreHandleType__enumvalues = hipExternalSemaphoreHandleType_enum__enumvalues
class struct_hipExternalSemaphoreHandleDesc_st(Structure):
    pass

class union_hipExternalSemaphoreHandleDesc_st_handle(Union):
    pass

class struct_hipExternalSemaphoreHandleDesc_st_0_win32(Structure):
    pass

struct_hipExternalSemaphoreHandleDesc_st_0_win32._pack_ = 1 # source:False
struct_hipExternalSemaphoreHandleDesc_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_hipExternalSemaphoreHandleDesc_st_handle._pack_ = 1 # source:False
union_hipExternalSemaphoreHandleDesc_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_hipExternalSemaphoreHandleDesc_st_0_win32),
    ('NvSciSyncObj', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_hipExternalSemaphoreHandleDesc_st._pack_ = 1 # source:False
struct_hipExternalSemaphoreHandleDesc_st._fields_ = [
    ('type', hipExternalSemaphoreHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_hipExternalSemaphoreHandleDesc_st_handle),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

hipExternalSemaphoreHandleDesc = struct_hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t = ctypes.POINTER(None)
class struct_hipExternalSemaphoreSignalParams_st(Structure):
    pass

class struct_hipExternalSemaphoreSignalParams_st_params(Structure):
    pass

class struct_hipExternalSemaphoreSignalParams_st_0_fence(Structure):
    pass

struct_hipExternalSemaphoreSignalParams_st_0_fence._pack_ = 1 # source:False
struct_hipExternalSemaphoreSignalParams_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_hipExternalSemaphoreSignalParams_st_0_nvSciSync(Union):
    pass

union_hipExternalSemaphoreSignalParams_st_0_nvSciSync._pack_ = 1 # source:False
union_hipExternalSemaphoreSignalParams_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_hipExternalSemaphoreSignalParams_st_0_keyedMutex(Structure):
    pass

struct_hipExternalSemaphoreSignalParams_st_0_keyedMutex._pack_ = 1 # source:False
struct_hipExternalSemaphoreSignalParams_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
]

struct_hipExternalSemaphoreSignalParams_st_params._pack_ = 1 # source:False
struct_hipExternalSemaphoreSignalParams_st_params._fields_ = [
    ('fence', struct_hipExternalSemaphoreSignalParams_st_0_fence),
    ('nvSciSync', union_hipExternalSemaphoreSignalParams_st_0_nvSciSync),
    ('keyedMutex', struct_hipExternalSemaphoreSignalParams_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 12),
]

struct_hipExternalSemaphoreSignalParams_st._pack_ = 1 # source:False
struct_hipExternalSemaphoreSignalParams_st._fields_ = [
    ('params', struct_hipExternalSemaphoreSignalParams_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipExternalSemaphoreSignalParams = struct_hipExternalSemaphoreSignalParams_st
class struct_hipExternalSemaphoreWaitParams_st(Structure):
    pass

class struct_hipExternalSemaphoreWaitParams_st_params(Structure):
    pass

class struct_hipExternalSemaphoreWaitParams_st_0_fence(Structure):
    pass

struct_hipExternalSemaphoreWaitParams_st_0_fence._pack_ = 1 # source:False
struct_hipExternalSemaphoreWaitParams_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_hipExternalSemaphoreWaitParams_st_0_nvSciSync(Union):
    pass

union_hipExternalSemaphoreWaitParams_st_0_nvSciSync._pack_ = 1 # source:False
union_hipExternalSemaphoreWaitParams_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_hipExternalSemaphoreWaitParams_st_0_keyedMutex(Structure):
    pass

struct_hipExternalSemaphoreWaitParams_st_0_keyedMutex._pack_ = 1 # source:False
struct_hipExternalSemaphoreWaitParams_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
    ('timeoutMs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_hipExternalSemaphoreWaitParams_st_params._pack_ = 1 # source:False
struct_hipExternalSemaphoreWaitParams_st_params._fields_ = [
    ('fence', struct_hipExternalSemaphoreWaitParams_st_0_fence),
    ('nvSciSync', union_hipExternalSemaphoreWaitParams_st_0_nvSciSync),
    ('keyedMutex', struct_hipExternalSemaphoreWaitParams_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 10),
]

struct_hipExternalSemaphoreWaitParams_st._pack_ = 1 # source:False
struct_hipExternalSemaphoreWaitParams_st._fields_ = [
    ('params', struct_hipExternalSemaphoreWaitParams_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipExternalSemaphoreWaitParams = struct_hipExternalSemaphoreWaitParams_st
try:
    __hipGetPCH = _libraries['libamdhip64.so'].__hipGetPCH
    __hipGetPCH.restype = None
    __hipGetPCH.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass

# values for enumeration 'hipGraphicsRegisterFlags'
hipGraphicsRegisterFlags__enumvalues = {
    0: 'hipGraphicsRegisterFlagsNone',
    1: 'hipGraphicsRegisterFlagsReadOnly',
    2: 'hipGraphicsRegisterFlagsWriteDiscard',
    4: 'hipGraphicsRegisterFlagsSurfaceLoadStore',
    8: 'hipGraphicsRegisterFlagsTextureGather',
}
hipGraphicsRegisterFlagsNone = 0
hipGraphicsRegisterFlagsReadOnly = 1
hipGraphicsRegisterFlagsWriteDiscard = 2
hipGraphicsRegisterFlagsSurfaceLoadStore = 4
hipGraphicsRegisterFlagsTextureGather = 8
hipGraphicsRegisterFlags = ctypes.c_uint32 # enum
class struct__hipGraphicsResource(Structure):
    pass

hipGraphicsResource = struct__hipGraphicsResource
hipGraphicsResource_t = ctypes.POINTER(struct__hipGraphicsResource)
class struct_ihipGraph(Structure):
    pass

hipGraph_t = ctypes.POINTER(struct_ihipGraph)
class struct_hipGraphNode(Structure):
    pass

hipGraphNode_t = ctypes.POINTER(struct_hipGraphNode)
class struct_hipGraphExec(Structure):
    pass

hipGraphExec_t = ctypes.POINTER(struct_hipGraphExec)
class struct_hipUserObject(Structure):
    pass

hipUserObject_t = ctypes.POINTER(struct_hipUserObject)

# values for enumeration 'hipGraphNodeType'
hipGraphNodeType__enumvalues = {
    0: 'hipGraphNodeTypeKernel',
    1: 'hipGraphNodeTypeMemcpy',
    2: 'hipGraphNodeTypeMemset',
    3: 'hipGraphNodeTypeHost',
    4: 'hipGraphNodeTypeGraph',
    5: 'hipGraphNodeTypeEmpty',
    6: 'hipGraphNodeTypeWaitEvent',
    7: 'hipGraphNodeTypeEventRecord',
    8: 'hipGraphNodeTypeExtSemaphoreSignal',
    9: 'hipGraphNodeTypeExtSemaphoreWait',
    10: 'hipGraphNodeTypeMemAlloc',
    11: 'hipGraphNodeTypeMemFree',
    12: 'hipGraphNodeTypeMemcpyFromSymbol',
    13: 'hipGraphNodeTypeMemcpyToSymbol',
    14: 'hipGraphNodeTypeCount',
}
hipGraphNodeTypeKernel = 0
hipGraphNodeTypeMemcpy = 1
hipGraphNodeTypeMemset = 2
hipGraphNodeTypeHost = 3
hipGraphNodeTypeGraph = 4
hipGraphNodeTypeEmpty = 5
hipGraphNodeTypeWaitEvent = 6
hipGraphNodeTypeEventRecord = 7
hipGraphNodeTypeExtSemaphoreSignal = 8
hipGraphNodeTypeExtSemaphoreWait = 9
hipGraphNodeTypeMemAlloc = 10
hipGraphNodeTypeMemFree = 11
hipGraphNodeTypeMemcpyFromSymbol = 12
hipGraphNodeTypeMemcpyToSymbol = 13
hipGraphNodeTypeCount = 14
hipGraphNodeType = ctypes.c_uint32 # enum
hipHostFn_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))
class struct_hipHostNodeParams(Structure):
    pass

struct_hipHostNodeParams._pack_ = 1 # source:False
struct_hipHostNodeParams._fields_ = [
    ('fn', ctypes.CFUNCTYPE(None, ctypes.POINTER(None))),
    ('userData', ctypes.POINTER(None)),
]

hipHostNodeParams = struct_hipHostNodeParams
class struct_hipKernelNodeParams(Structure):
    pass

struct_hipKernelNodeParams._pack_ = 1 # source:False
struct_hipKernelNodeParams._fields_ = [
    ('blockDim', dim3),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('extra', ctypes.POINTER(ctypes.POINTER(None))),
    ('func', ctypes.POINTER(None)),
    ('gridDim', dim3),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

hipKernelNodeParams = struct_hipKernelNodeParams
class struct_hipMemsetParams(Structure):
    pass

struct_hipMemsetParams._pack_ = 1 # source:False
struct_hipMemsetParams._fields_ = [
    ('dst', ctypes.POINTER(None)),
    ('elementSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('height', ctypes.c_uint64),
    ('pitch', ctypes.c_uint64),
    ('value', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
]

hipMemsetParams = struct_hipMemsetParams
class struct_hipMemAllocNodeParams(Structure):
    pass

struct_hipMemAllocNodeParams._pack_ = 1 # source:False
struct_hipMemAllocNodeParams._fields_ = [
    ('poolProps', hipMemPoolProps),
    ('accessDescs', ctypes.POINTER(struct_hipMemAccessDesc)),
    ('accessDescCount', ctypes.c_uint64),
    ('bytesize', ctypes.c_uint64),
    ('dptr', ctypes.POINTER(None)),
]

hipMemAllocNodeParams = struct_hipMemAllocNodeParams

# values for enumeration 'hipKernelNodeAttrID'
hipKernelNodeAttrID__enumvalues = {
    1: 'hipKernelNodeAttributeAccessPolicyWindow',
    2: 'hipKernelNodeAttributeCooperative',
}
hipKernelNodeAttributeAccessPolicyWindow = 1
hipKernelNodeAttributeCooperative = 2
hipKernelNodeAttrID = ctypes.c_uint32 # enum

# values for enumeration 'hipAccessProperty'
hipAccessProperty__enumvalues = {
    0: 'hipAccessPropertyNormal',
    1: 'hipAccessPropertyStreaming',
    2: 'hipAccessPropertyPersisting',
}
hipAccessPropertyNormal = 0
hipAccessPropertyStreaming = 1
hipAccessPropertyPersisting = 2
hipAccessProperty = ctypes.c_uint32 # enum
class struct_hipAccessPolicyWindow(Structure):
    pass

struct_hipAccessPolicyWindow._pack_ = 1 # source:False
struct_hipAccessPolicyWindow._fields_ = [
    ('base_ptr', ctypes.POINTER(None)),
    ('hitProp', hipAccessProperty),
    ('hitRatio', ctypes.c_float),
    ('missProp', hipAccessProperty),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('num_bytes', ctypes.c_uint64),
]

hipAccessPolicyWindow = struct_hipAccessPolicyWindow
class union_hipKernelNodeAttrValue(Union):
    pass

union_hipKernelNodeAttrValue._pack_ = 1 # source:False
union_hipKernelNodeAttrValue._fields_ = [
    ('accessPolicyWindow', hipAccessPolicyWindow),
    ('cooperative', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

hipKernelNodeAttrValue = union_hipKernelNodeAttrValue

# values for enumeration 'hipGraphExecUpdateResult'
hipGraphExecUpdateResult__enumvalues = {
    0: 'hipGraphExecUpdateSuccess',
    1: 'hipGraphExecUpdateError',
    2: 'hipGraphExecUpdateErrorTopologyChanged',
    3: 'hipGraphExecUpdateErrorNodeTypeChanged',
    4: 'hipGraphExecUpdateErrorFunctionChanged',
    5: 'hipGraphExecUpdateErrorParametersChanged',
    6: 'hipGraphExecUpdateErrorNotSupported',
    7: 'hipGraphExecUpdateErrorUnsupportedFunctionChange',
}
hipGraphExecUpdateSuccess = 0
hipGraphExecUpdateError = 1
hipGraphExecUpdateErrorTopologyChanged = 2
hipGraphExecUpdateErrorNodeTypeChanged = 3
hipGraphExecUpdateErrorFunctionChanged = 4
hipGraphExecUpdateErrorParametersChanged = 5
hipGraphExecUpdateErrorNotSupported = 6
hipGraphExecUpdateErrorUnsupportedFunctionChange = 7
hipGraphExecUpdateResult = ctypes.c_uint32 # enum

# values for enumeration 'hipStreamCaptureMode'
hipStreamCaptureMode__enumvalues = {
    0: 'hipStreamCaptureModeGlobal',
    1: 'hipStreamCaptureModeThreadLocal',
    2: 'hipStreamCaptureModeRelaxed',
}
hipStreamCaptureModeGlobal = 0
hipStreamCaptureModeThreadLocal = 1
hipStreamCaptureModeRelaxed = 2
hipStreamCaptureMode = ctypes.c_uint32 # enum

# values for enumeration 'hipStreamCaptureStatus'
hipStreamCaptureStatus__enumvalues = {
    0: 'hipStreamCaptureStatusNone',
    1: 'hipStreamCaptureStatusActive',
    2: 'hipStreamCaptureStatusInvalidated',
}
hipStreamCaptureStatusNone = 0
hipStreamCaptureStatusActive = 1
hipStreamCaptureStatusInvalidated = 2
hipStreamCaptureStatus = ctypes.c_uint32 # enum

# values for enumeration 'hipStreamUpdateCaptureDependenciesFlags'
hipStreamUpdateCaptureDependenciesFlags__enumvalues = {
    0: 'hipStreamAddCaptureDependencies',
    1: 'hipStreamSetCaptureDependencies',
}
hipStreamAddCaptureDependencies = 0
hipStreamSetCaptureDependencies = 1
hipStreamUpdateCaptureDependenciesFlags = ctypes.c_uint32 # enum

# values for enumeration 'hipGraphMemAttributeType'
hipGraphMemAttributeType__enumvalues = {
    0: 'hipGraphMemAttrUsedMemCurrent',
    1: 'hipGraphMemAttrUsedMemHigh',
    2: 'hipGraphMemAttrReservedMemCurrent',
    3: 'hipGraphMemAttrReservedMemHigh',
}
hipGraphMemAttrUsedMemCurrent = 0
hipGraphMemAttrUsedMemHigh = 1
hipGraphMemAttrReservedMemCurrent = 2
hipGraphMemAttrReservedMemHigh = 3
hipGraphMemAttributeType = ctypes.c_uint32 # enum

# values for enumeration 'hipUserObjectFlags'
hipUserObjectFlags__enumvalues = {
    1: 'hipUserObjectNoDestructorSync',
}
hipUserObjectNoDestructorSync = 1
hipUserObjectFlags = ctypes.c_uint32 # enum

# values for enumeration 'hipUserObjectRetainFlags'
hipUserObjectRetainFlags__enumvalues = {
    1: 'hipGraphUserObjectMove',
}
hipGraphUserObjectMove = 1
hipUserObjectRetainFlags = ctypes.c_uint32 # enum

# values for enumeration 'hipGraphInstantiateFlags'
hipGraphInstantiateFlags__enumvalues = {
    1: 'hipGraphInstantiateFlagAutoFreeOnLaunch',
    2: 'hipGraphInstantiateFlagUpload',
    4: 'hipGraphInstantiateFlagDeviceLaunch',
    8: 'hipGraphInstantiateFlagUseNodePriority',
}
hipGraphInstantiateFlagAutoFreeOnLaunch = 1
hipGraphInstantiateFlagUpload = 2
hipGraphInstantiateFlagDeviceLaunch = 4
hipGraphInstantiateFlagUseNodePriority = 8
hipGraphInstantiateFlags = ctypes.c_uint32 # enum

# values for enumeration 'hipGraphDebugDotFlags'
hipGraphDebugDotFlags__enumvalues = {
    1: 'hipGraphDebugDotFlagsVerbose',
    4: 'hipGraphDebugDotFlagsKernelNodeParams',
    8: 'hipGraphDebugDotFlagsMemcpyNodeParams',
    16: 'hipGraphDebugDotFlagsMemsetNodeParams',
    32: 'hipGraphDebugDotFlagsHostNodeParams',
    64: 'hipGraphDebugDotFlagsEventNodeParams',
    128: 'hipGraphDebugDotFlagsExtSemasSignalNodeParams',
    256: 'hipGraphDebugDotFlagsExtSemasWaitNodeParams',
    512: 'hipGraphDebugDotFlagsKernelNodeAttributes',
    1024: 'hipGraphDebugDotFlagsHandles',
}
hipGraphDebugDotFlagsVerbose = 1
hipGraphDebugDotFlagsKernelNodeParams = 4
hipGraphDebugDotFlagsMemcpyNodeParams = 8
hipGraphDebugDotFlagsMemsetNodeParams = 16
hipGraphDebugDotFlagsHostNodeParams = 32
hipGraphDebugDotFlagsEventNodeParams = 64
hipGraphDebugDotFlagsExtSemasSignalNodeParams = 128
hipGraphDebugDotFlagsExtSemasWaitNodeParams = 256
hipGraphDebugDotFlagsKernelNodeAttributes = 512
hipGraphDebugDotFlagsHandles = 1024
hipGraphDebugDotFlags = ctypes.c_uint32 # enum
class struct_hipMemAllocationProp(Structure):
    pass

class struct_hipMemAllocationProp_allocFlags(Structure):
    pass

struct_hipMemAllocationProp_allocFlags._pack_ = 1 # source:False
struct_hipMemAllocationProp_allocFlags._fields_ = [
    ('compressionType', ctypes.c_ubyte),
    ('gpuDirectRDMACapable', ctypes.c_ubyte),
    ('usage', ctypes.c_uint16),
]

struct_hipMemAllocationProp._pack_ = 1 # source:False
struct_hipMemAllocationProp._fields_ = [
    ('type', hipMemAllocationType),
    ('requestedHandleType', hipMemAllocationHandleType),
    ('location', hipMemLocation),
    ('win32HandleMetaData', ctypes.POINTER(None)),
    ('allocFlags', struct_hipMemAllocationProp_allocFlags),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipMemAllocationProp = struct_hipMemAllocationProp
class struct_hipExternalSemaphoreSignalNodeParams(Structure):
    pass

struct_hipExternalSemaphoreSignalNodeParams._pack_ = 1 # source:False
struct_hipExternalSemaphoreSignalNodeParams._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(None))),
    ('paramsArray', ctypes.POINTER(struct_hipExternalSemaphoreSignalParams_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipExternalSemaphoreSignalNodeParams = struct_hipExternalSemaphoreSignalNodeParams
class struct_hipExternalSemaphoreWaitNodeParams(Structure):
    pass

struct_hipExternalSemaphoreWaitNodeParams._pack_ = 1 # source:False
struct_hipExternalSemaphoreWaitNodeParams._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(None))),
    ('paramsArray', ctypes.POINTER(struct_hipExternalSemaphoreWaitParams_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hipExternalSemaphoreWaitNodeParams = struct_hipExternalSemaphoreWaitNodeParams
class struct_ihipMemGenericAllocationHandle(Structure):
    pass

hipMemGenericAllocationHandle_t = ctypes.POINTER(struct_ihipMemGenericAllocationHandle)

# values for enumeration 'hipMemAllocationGranularity_flags'
hipMemAllocationGranularity_flags__enumvalues = {
    0: 'hipMemAllocationGranularityMinimum',
    1: 'hipMemAllocationGranularityRecommended',
}
hipMemAllocationGranularityMinimum = 0
hipMemAllocationGranularityRecommended = 1
hipMemAllocationGranularity_flags = ctypes.c_uint32 # enum

# values for enumeration 'hipMemHandleType'
hipMemHandleType__enumvalues = {
    0: 'hipMemHandleTypeGeneric',
}
hipMemHandleTypeGeneric = 0
hipMemHandleType = ctypes.c_uint32 # enum

# values for enumeration 'hipMemOperationType'
hipMemOperationType__enumvalues = {
    1: 'hipMemOperationTypeMap',
    2: 'hipMemOperationTypeUnmap',
}
hipMemOperationTypeMap = 1
hipMemOperationTypeUnmap = 2
hipMemOperationType = ctypes.c_uint32 # enum

# values for enumeration 'hipArraySparseSubresourceType'
hipArraySparseSubresourceType__enumvalues = {
    0: 'hipArraySparseSubresourceTypeSparseLevel',
    1: 'hipArraySparseSubresourceTypeMiptail',
}
hipArraySparseSubresourceTypeSparseLevel = 0
hipArraySparseSubresourceTypeMiptail = 1
hipArraySparseSubresourceType = ctypes.c_uint32 # enum
class struct_hipArrayMapInfo(Structure):
    pass

class union_hipArrayMapInfo_resource(Union):
    pass

union_hipArrayMapInfo_resource._pack_ = 1 # source:False
union_hipArrayMapInfo_resource._fields_ = [
    ('mipmap', hipMipmappedArray),
    ('array', ctypes.POINTER(struct_hipArray)),
    ('PADDING_0', ctypes.c_ubyte * 56),
]

class union_hipArrayMapInfo_subresource(Union):
    pass

class struct_hipArrayMapInfo_1_sparseLevel(Structure):
    pass

struct_hipArrayMapInfo_1_sparseLevel._pack_ = 1 # source:False
struct_hipArrayMapInfo_1_sparseLevel._fields_ = [
    ('level', ctypes.c_uint32),
    ('layer', ctypes.c_uint32),
    ('offsetX', ctypes.c_uint32),
    ('offsetY', ctypes.c_uint32),
    ('offsetZ', ctypes.c_uint32),
    ('extentWidth', ctypes.c_uint32),
    ('extentHeight', ctypes.c_uint32),
    ('extentDepth', ctypes.c_uint32),
]

class struct_hipArrayMapInfo_1_miptail(Structure):
    pass

struct_hipArrayMapInfo_1_miptail._pack_ = 1 # source:False
struct_hipArrayMapInfo_1_miptail._fields_ = [
    ('layer', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

union_hipArrayMapInfo_subresource._pack_ = 1 # source:False
union_hipArrayMapInfo_subresource._fields_ = [
    ('sparseLevel', struct_hipArrayMapInfo_1_sparseLevel),
    ('miptail', struct_hipArrayMapInfo_1_miptail),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

class union_hipArrayMapInfo_memHandle(Union):
    pass

union_hipArrayMapInfo_memHandle._pack_ = 1 # source:False
union_hipArrayMapInfo_memHandle._fields_ = [
    ('memHandle', ctypes.POINTER(struct_ihipMemGenericAllocationHandle)),
]

struct_hipArrayMapInfo._pack_ = 1 # source:False
struct_hipArrayMapInfo._fields_ = [
    ('resourceType', hipResourceType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('resource', union_hipArrayMapInfo_resource),
    ('subresourceType', hipArraySparseSubresourceType),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('subresource', union_hipArrayMapInfo_subresource),
    ('memOperationType', hipMemOperationType),
    ('memHandleType', hipMemHandleType),
    ('memHandle', union_hipArrayMapInfo_memHandle),
    ('offset', ctypes.c_uint64),
    ('deviceBitMask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 2),
]

hipArrayMapInfo = struct_hipArrayMapInfo
try:
    hipInit = _libraries['libamdhip64.so'].hipInit
    hipInit.restype = hipError_t
    hipInit.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipDriverGetVersion = _libraries['libamdhip64.so'].hipDriverGetVersion
    hipDriverGetVersion.restype = hipError_t
    hipDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipRuntimeGetVersion = _libraries['libamdhip64.so'].hipRuntimeGetVersion
    hipRuntimeGetVersion.restype = hipError_t
    hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipDeviceGet = _libraries['libamdhip64.so'].hipDeviceGet
    hipDeviceGet.restype = hipError_t
    hipDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceComputeCapability = _libraries['libamdhip64.so'].hipDeviceComputeCapability
    hipDeviceComputeCapability.restype = hipError_t
    hipDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipDevice_t]
except AttributeError:
    pass
try:
    hipDeviceGetName = _libraries['libamdhip64.so'].hipDeviceGetName
    hipDeviceGetName.restype = hipError_t
    hipDeviceGetName.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, hipDevice_t]
except AttributeError:
    pass
try:
    hipDeviceGetUuid = _libraries['libamdhip64.so'].hipDeviceGetUuid
    hipDeviceGetUuid.restype = hipError_t
    hipDeviceGetUuid.argtypes = [ctypes.POINTER(struct_hipUUID_t), hipDevice_t]
except AttributeError:
    pass
try:
    hipDeviceGetP2PAttribute = _libraries['libamdhip64.so'].hipDeviceGetP2PAttribute
    hipDeviceGetP2PAttribute.restype = hipError_t
    hipDeviceGetP2PAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), hipDeviceP2PAttr, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceGetPCIBusId = _libraries['libamdhip64.so'].hipDeviceGetPCIBusId
    hipDeviceGetPCIBusId.restype = hipError_t
    hipDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceGetByPCIBusId = _libraries['libamdhip64.so'].hipDeviceGetByPCIBusId
    hipDeviceGetByPCIBusId.restype = hipError_t
    hipDeviceGetByPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hipDeviceTotalMem = _libraries['libamdhip64.so'].hipDeviceTotalMem
    hipDeviceTotalMem.restype = hipError_t
    hipDeviceTotalMem.argtypes = [ctypes.POINTER(ctypes.c_uint64), hipDevice_t]
except AttributeError:
    pass
try:
    hipDeviceSynchronize = _libraries['libamdhip64.so'].hipDeviceSynchronize
    hipDeviceSynchronize.restype = hipError_t
    hipDeviceSynchronize.argtypes = []
except AttributeError:
    pass
try:
    hipDeviceReset = _libraries['libamdhip64.so'].hipDeviceReset
    hipDeviceReset.restype = hipError_t
    hipDeviceReset.argtypes = []
except AttributeError:
    pass
try:
    hipSetDevice = _libraries['libamdhip64.so'].hipSetDevice
    hipSetDevice.restype = hipError_t
    hipSetDevice.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    hipGetDevice = _libraries['libamdhip64.so'].hipGetDevice
    hipGetDevice.restype = hipError_t
    hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipGetDeviceCount = _libraries['libamdhip64.so'].hipGetDeviceCount
    hipGetDeviceCount.restype = hipError_t
    hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipDeviceGetAttribute = _libraries['libamdhip64.so'].hipDeviceGetAttribute
    hipDeviceGetAttribute.restype = hipError_t
    hipDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), hipDeviceAttribute_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceGetDefaultMemPool = _libraries['libamdhip64.so'].hipDeviceGetDefaultMemPool
    hipDeviceGetDefaultMemPool.restype = hipError_t
    hipDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemPoolHandle_t)), ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceSetMemPool = _libraries['libamdhip64.so'].hipDeviceSetMemPool
    hipDeviceSetMemPool.restype = hipError_t
    hipDeviceSetMemPool.argtypes = [ctypes.c_int32, hipMemPool_t]
except AttributeError:
    pass
try:
    hipDeviceGetMemPool = _libraries['libamdhip64.so'].hipDeviceGetMemPool
    hipDeviceGetMemPool.restype = hipError_t
    hipDeviceGetMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemPoolHandle_t)), ctypes.c_int32]
except AttributeError:
    pass
try:
    hipGetDevicePropertiesR0600 = _libraries['libamdhip64.so'].hipGetDevicePropertiesR0600
    hipGetDevicePropertiesR0600.restype = hipError_t
    hipGetDevicePropertiesR0600.argtypes = [ctypes.POINTER(struct_hipDeviceProp_tR0600), ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceSetCacheConfig = _libraries['libamdhip64.so'].hipDeviceSetCacheConfig
    hipDeviceSetCacheConfig.restype = hipError_t
    hipDeviceSetCacheConfig.argtypes = [hipFuncCache_t]
except AttributeError:
    pass
try:
    hipDeviceGetCacheConfig = _libraries['libamdhip64.so'].hipDeviceGetCacheConfig
    hipDeviceGetCacheConfig.restype = hipError_t
    hipDeviceGetCacheConfig.argtypes = [ctypes.POINTER(hipFuncCache_t)]
except AttributeError:
    pass
try:
    hipDeviceGetLimit = _libraries['libamdhip64.so'].hipDeviceGetLimit
    hipDeviceGetLimit.restype = hipError_t
    hipDeviceGetLimit.argtypes = [ctypes.POINTER(ctypes.c_uint64), hipLimit_t]
except AttributeError:
    pass
try:
    hipDeviceSetLimit = _libraries['libamdhip64.so'].hipDeviceSetLimit
    hipDeviceSetLimit.restype = hipError_t
    hipDeviceSetLimit.argtypes = [hipLimit_t, size_t]
except AttributeError:
    pass
try:
    hipDeviceGetSharedMemConfig = _libraries['libamdhip64.so'].hipDeviceGetSharedMemConfig
    hipDeviceGetSharedMemConfig.restype = hipError_t
    hipDeviceGetSharedMemConfig.argtypes = [ctypes.POINTER(hipSharedMemConfig)]
except AttributeError:
    pass
try:
    hipGetDeviceFlags = _libraries['libamdhip64.so'].hipGetDeviceFlags
    hipGetDeviceFlags.restype = hipError_t
    hipGetDeviceFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipDeviceSetSharedMemConfig = _libraries['libamdhip64.so'].hipDeviceSetSharedMemConfig
    hipDeviceSetSharedMemConfig.restype = hipError_t
    hipDeviceSetSharedMemConfig.argtypes = [hipSharedMemConfig]
except AttributeError:
    pass
try:
    hipSetDeviceFlags = _libraries['libamdhip64.so'].hipSetDeviceFlags
    hipSetDeviceFlags.restype = hipError_t
    hipSetDeviceFlags.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipChooseDeviceR0600 = _libraries['libamdhip64.so'].hipChooseDeviceR0600
    hipChooseDeviceR0600.restype = hipError_t
    hipChooseDeviceR0600.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(struct_hipDeviceProp_tR0600)]
except AttributeError:
    pass
try:
    hipExtGetLinkTypeAndHopCount = _libraries['libamdhip64.so'].hipExtGetLinkTypeAndHopCount
    hipExtGetLinkTypeAndHopCount.restype = hipError_t
    hipExtGetLinkTypeAndHopCount.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipIpcGetMemHandle = _libraries['libamdhip64.so'].hipIpcGetMemHandle
    hipIpcGetMemHandle.restype = hipError_t
    hipIpcGetMemHandle.argtypes = [ctypes.POINTER(struct_hipIpcMemHandle_st), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipIpcOpenMemHandle = _libraries['libamdhip64.so'].hipIpcOpenMemHandle
    hipIpcOpenMemHandle.restype = hipError_t
    hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), hipIpcMemHandle_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipIpcCloseMemHandle = _libraries['libamdhip64.so'].hipIpcCloseMemHandle
    hipIpcCloseMemHandle.restype = hipError_t
    hipIpcCloseMemHandle.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipIpcGetEventHandle = _libraries['libamdhip64.so'].hipIpcGetEventHandle
    hipIpcGetEventHandle.restype = hipError_t
    hipIpcGetEventHandle.argtypes = [ctypes.POINTER(struct_hipIpcEventHandle_st), hipEvent_t]
except AttributeError:
    pass
try:
    hipIpcOpenEventHandle = _libraries['libamdhip64.so'].hipIpcOpenEventHandle
    hipIpcOpenEventHandle.restype = hipError_t
    hipIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipEvent_t)), hipIpcEventHandle_t]
except AttributeError:
    pass
try:
    hipFuncSetAttribute = _libraries['libamdhip64.so'].hipFuncSetAttribute
    hipFuncSetAttribute.restype = hipError_t
    hipFuncSetAttribute.argtypes = [ctypes.POINTER(None), hipFuncAttribute, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipFuncSetCacheConfig = _libraries['libamdhip64.so'].hipFuncSetCacheConfig
    hipFuncSetCacheConfig.restype = hipError_t
    hipFuncSetCacheConfig.argtypes = [ctypes.POINTER(None), hipFuncCache_t]
except AttributeError:
    pass
try:
    hipFuncSetSharedMemConfig = _libraries['libamdhip64.so'].hipFuncSetSharedMemConfig
    hipFuncSetSharedMemConfig.restype = hipError_t
    hipFuncSetSharedMemConfig.argtypes = [ctypes.POINTER(None), hipSharedMemConfig]
except AttributeError:
    pass
try:
    hipGetLastError = _libraries['libamdhip64.so'].hipGetLastError
    hipGetLastError.restype = hipError_t
    hipGetLastError.argtypes = []
except AttributeError:
    pass
try:
    hipExtGetLastError = _libraries['libamdhip64.so'].hipExtGetLastError
    hipExtGetLastError.restype = hipError_t
    hipExtGetLastError.argtypes = []
except AttributeError:
    pass
try:
    hipPeekAtLastError = _libraries['libamdhip64.so'].hipPeekAtLastError
    hipPeekAtLastError.restype = hipError_t
    hipPeekAtLastError.argtypes = []
except AttributeError:
    pass
try:
    hipGetErrorName = _libraries['libamdhip64.so'].hipGetErrorName
    hipGetErrorName.restype = ctypes.POINTER(ctypes.c_char)
    hipGetErrorName.argtypes = [hipError_t]
except AttributeError:
    pass
try:
    hipGetErrorString = _libraries['libamdhip64.so'].hipGetErrorString
    hipGetErrorString.restype = ctypes.POINTER(ctypes.c_char)
    hipGetErrorString.argtypes = [hipError_t]
except AttributeError:
    pass
try:
    hipDrvGetErrorName = _libraries['libamdhip64.so'].hipDrvGetErrorName
    hipDrvGetErrorName.restype = hipError_t
    hipDrvGetErrorName.argtypes = [hipError_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hipDrvGetErrorString = _libraries['libamdhip64.so'].hipDrvGetErrorString
    hipDrvGetErrorString.restype = hipError_t
    hipDrvGetErrorString.argtypes = [hipError_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hipStreamCreate = _libraries['libamdhip64.so'].hipStreamCreate
    hipStreamCreate.restype = hipError_t
    hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipStream_t))]
except AttributeError:
    pass
try:
    hipStreamCreateWithFlags = _libraries['libamdhip64.so'].hipStreamCreateWithFlags
    hipStreamCreateWithFlags.restype = hipError_t
    hipStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipStream_t)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipStreamCreateWithPriority = _libraries['libamdhip64.so'].hipStreamCreateWithPriority
    hipStreamCreateWithPriority.restype = hipError_t
    hipStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipStream_t)), ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceGetStreamPriorityRange = _libraries['libamdhip64.so'].hipDeviceGetStreamPriorityRange
    hipDeviceGetStreamPriorityRange.restype = hipError_t
    hipDeviceGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipStreamDestroy = _libraries['libamdhip64.so'].hipStreamDestroy
    hipStreamDestroy.restype = hipError_t
    hipStreamDestroy.argtypes = [hipStream_t]
except AttributeError:
    pass
try:
    hipStreamQuery = _libraries['libamdhip64.so'].hipStreamQuery
    hipStreamQuery.restype = hipError_t
    hipStreamQuery.argtypes = [hipStream_t]
except AttributeError:
    pass
try:
    hipStreamSynchronize = _libraries['libamdhip64.so'].hipStreamSynchronize
    hipStreamSynchronize.restype = hipError_t
    hipStreamSynchronize.argtypes = [hipStream_t]
except AttributeError:
    pass
try:
    hipStreamWaitEvent = _libraries['libamdhip64.so'].hipStreamWaitEvent
    hipStreamWaitEvent.restype = hipError_t
    hipStreamWaitEvent.argtypes = [hipStream_t, hipEvent_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipStreamGetFlags = _libraries['libamdhip64.so'].hipStreamGetFlags
    hipStreamGetFlags.restype = hipError_t
    hipStreamGetFlags.argtypes = [hipStream_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipStreamGetPriority = _libraries['libamdhip64.so'].hipStreamGetPriority
    hipStreamGetPriority.restype = hipError_t
    hipStreamGetPriority.argtypes = [hipStream_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipStreamGetDevice = _libraries['libamdhip64.so'].hipStreamGetDevice
    hipStreamGetDevice.restype = hipError_t
    hipStreamGetDevice.argtypes = [hipStream_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    hipExtStreamCreateWithCUMask = _libraries['libamdhip64.so'].hipExtStreamCreateWithCUMask
    hipExtStreamCreateWithCUMask.restype = hipError_t
    hipExtStreamCreateWithCUMask.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipStream_t)), uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipExtStreamGetCUMask = _libraries['libamdhip64.so'].hipExtStreamGetCUMask
    hipExtStreamGetCUMask.restype = hipError_t
    hipExtStreamGetCUMask.argtypes = [hipStream_t, uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
hipStreamCallback_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ihipStream_t), hipError_t, ctypes.POINTER(None))
try:
    hipStreamAddCallback = _libraries['libamdhip64.so'].hipStreamAddCallback
    hipStreamAddCallback.restype = hipError_t
    hipStreamAddCallback.argtypes = [hipStream_t, hipStreamCallback_t, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipStreamWaitValue32 = _libraries['libamdhip64.so'].hipStreamWaitValue32
    hipStreamWaitValue32.restype = hipError_t
    hipStreamWaitValue32.argtypes = [hipStream_t, ctypes.POINTER(None), uint32_t, ctypes.c_uint32, uint32_t]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    hipStreamWaitValue64 = _libraries['libamdhip64.so'].hipStreamWaitValue64
    hipStreamWaitValue64.restype = hipError_t
    hipStreamWaitValue64.argtypes = [hipStream_t, ctypes.POINTER(None), uint64_t, ctypes.c_uint32, uint64_t]
except AttributeError:
    pass
try:
    hipStreamWriteValue32 = _libraries['libamdhip64.so'].hipStreamWriteValue32
    hipStreamWriteValue32.restype = hipError_t
    hipStreamWriteValue32.argtypes = [hipStream_t, ctypes.POINTER(None), uint32_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipStreamWriteValue64 = _libraries['libamdhip64.so'].hipStreamWriteValue64
    hipStreamWriteValue64.restype = hipError_t
    hipStreamWriteValue64.argtypes = [hipStream_t, ctypes.POINTER(None), uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipEventCreateWithFlags = _libraries['libamdhip64.so'].hipEventCreateWithFlags
    hipEventCreateWithFlags.restype = hipError_t
    hipEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipEvent_t)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipEventCreate = _libraries['libamdhip64.so'].hipEventCreate
    hipEventCreate.restype = hipError_t
    hipEventCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipEvent_t))]
except AttributeError:
    pass
try:
    hipEventRecord = _libraries['libamdhip64.so'].hipEventRecord
    hipEventRecord.restype = hipError_t
    hipEventRecord.argtypes = [hipEvent_t, hipStream_t]
except AttributeError:
    pass
try:
    hipEventDestroy = _libraries['libamdhip64.so'].hipEventDestroy
    hipEventDestroy.restype = hipError_t
    hipEventDestroy.argtypes = [hipEvent_t]
except AttributeError:
    pass
try:
    hipEventSynchronize = _libraries['libamdhip64.so'].hipEventSynchronize
    hipEventSynchronize.restype = hipError_t
    hipEventSynchronize.argtypes = [hipEvent_t]
except AttributeError:
    pass
try:
    hipEventElapsedTime = _libraries['libamdhip64.so'].hipEventElapsedTime
    hipEventElapsedTime.restype = hipError_t
    hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), hipEvent_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipEventQuery = _libraries['libamdhip64.so'].hipEventQuery
    hipEventQuery.restype = hipError_t
    hipEventQuery.argtypes = [hipEvent_t]
except AttributeError:
    pass
try:
    hipPointerSetAttribute = _libraries['libamdhip64.so'].hipPointerSetAttribute
    hipPointerSetAttribute.restype = hipError_t
    hipPointerSetAttribute.argtypes = [ctypes.POINTER(None), hipPointer_attribute, hipDeviceptr_t]
except AttributeError:
    pass
try:
    hipPointerGetAttributes = _libraries['libamdhip64.so'].hipPointerGetAttributes
    hipPointerGetAttributes.restype = hipError_t
    hipPointerGetAttributes.argtypes = [ctypes.POINTER(struct_hipPointerAttribute_t), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipPointerGetAttribute = _libraries['libamdhip64.so'].hipPointerGetAttribute
    hipPointerGetAttribute.restype = hipError_t
    hipPointerGetAttribute.argtypes = [ctypes.POINTER(None), hipPointer_attribute, hipDeviceptr_t]
except AttributeError:
    pass
try:
    hipDrvPointerGetAttributes = _libraries['libamdhip64.so'].hipDrvPointerGetAttributes
    hipDrvPointerGetAttributes.restype = hipError_t
    hipDrvPointerGetAttributes.argtypes = [ctypes.c_uint32, ctypes.POINTER(hipPointer_attribute), ctypes.POINTER(ctypes.POINTER(None)), hipDeviceptr_t]
except AttributeError:
    pass
try:
    hipImportExternalSemaphore = _libraries['libamdhip64.so'].hipImportExternalSemaphore
    hipImportExternalSemaphore.restype = hipError_t
    hipImportExternalSemaphore.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_hipExternalSemaphoreHandleDesc_st)]
except AttributeError:
    pass
try:
    hipSignalExternalSemaphoresAsync = _libraries['libamdhip64.so'].hipSignalExternalSemaphoresAsync
    hipSignalExternalSemaphoresAsync.restype = hipError_t
    hipSignalExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_hipExternalSemaphoreSignalParams_st), ctypes.c_uint32, hipStream_t]
except AttributeError:
    pass
try:
    hipWaitExternalSemaphoresAsync = _libraries['libamdhip64.so'].hipWaitExternalSemaphoresAsync
    hipWaitExternalSemaphoresAsync.restype = hipError_t
    hipWaitExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_hipExternalSemaphoreWaitParams_st), ctypes.c_uint32, hipStream_t]
except AttributeError:
    pass
try:
    hipDestroyExternalSemaphore = _libraries['libamdhip64.so'].hipDestroyExternalSemaphore
    hipDestroyExternalSemaphore.restype = hipError_t
    hipDestroyExternalSemaphore.argtypes = [hipExternalSemaphore_t]
except AttributeError:
    pass
try:
    hipImportExternalMemory = _libraries['libamdhip64.so'].hipImportExternalMemory
    hipImportExternalMemory.restype = hipError_t
    hipImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_hipExternalMemoryHandleDesc_st)]
except AttributeError:
    pass
try:
    hipExternalMemoryGetMappedBuffer = _libraries['libamdhip64.so'].hipExternalMemoryGetMappedBuffer
    hipExternalMemoryGetMappedBuffer.restype = hipError_t
    hipExternalMemoryGetMappedBuffer.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), hipExternalMemory_t, ctypes.POINTER(struct_hipExternalMemoryBufferDesc_st)]
except AttributeError:
    pass
try:
    hipDestroyExternalMemory = _libraries['libamdhip64.so'].hipDestroyExternalMemory
    hipDestroyExternalMemory.restype = hipError_t
    hipDestroyExternalMemory.argtypes = [hipExternalMemory_t]
except AttributeError:
    pass
try:
    hipExternalMemoryGetMappedMipmappedArray = _libraries['FIXME_STUB'].hipExternalMemoryGetMappedMipmappedArray
    hipExternalMemoryGetMappedMipmappedArray.restype = hipError_t
    hipExternalMemoryGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipMipmappedArray)), hipExternalMemory_t, ctypes.POINTER(struct_hipExternalMemoryMipmappedArrayDesc_st)]
except AttributeError:
    pass
try:
    hipMalloc = _libraries['libamdhip64.so'].hipMalloc
    hipMalloc.restype = hipError_t
    hipMalloc.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t]
except AttributeError:
    pass
try:
    hipExtMallocWithFlags = _libraries['libamdhip64.so'].hipExtMallocWithFlags
    hipExtMallocWithFlags.restype = hipError_t
    hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMallocHost = _libraries['libamdhip64.so'].hipMallocHost
    hipMallocHost.restype = hipError_t
    hipMallocHost.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t]
except AttributeError:
    pass
try:
    hipMemAllocHost = _libraries['libamdhip64.so'].hipMemAllocHost
    hipMemAllocHost.restype = hipError_t
    hipMemAllocHost.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t]
except AttributeError:
    pass
try:
    hipHostMalloc = _libraries['libamdhip64.so'].hipHostMalloc
    hipHostMalloc.restype = hipError_t
    hipHostMalloc.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMallocManaged = _libraries['libamdhip64.so'].hipMallocManaged
    hipMallocManaged.restype = hipError_t
    hipMallocManaged.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMemPrefetchAsync = _libraries['libamdhip64.so'].hipMemPrefetchAsync
    hipMemPrefetchAsync.restype = hipError_t
    hipMemPrefetchAsync.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, hipStream_t]
except AttributeError:
    pass
try:
    hipMemAdvise = _libraries['libamdhip64.so'].hipMemAdvise
    hipMemAdvise.restype = hipError_t
    hipMemAdvise.argtypes = [ctypes.POINTER(None), size_t, hipMemoryAdvise, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipMemRangeGetAttribute = _libraries['libamdhip64.so'].hipMemRangeGetAttribute
    hipMemRangeGetAttribute.restype = hipError_t
    hipMemRangeGetAttribute.argtypes = [ctypes.POINTER(None), size_t, hipMemRangeAttribute, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipMemRangeGetAttributes = _libraries['libamdhip64.so'].hipMemRangeGetAttributes
    hipMemRangeGetAttributes.restype = hipError_t
    hipMemRangeGetAttributes.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(hipMemRangeAttribute), size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipStreamAttachMemAsync = _libraries['libamdhip64.so'].hipStreamAttachMemAsync
    hipStreamAttachMemAsync.restype = hipError_t
    hipStreamAttachMemAsync.argtypes = [hipStream_t, ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMallocAsync = _libraries['libamdhip64.so'].hipMallocAsync
    hipMallocAsync.restype = hipError_t
    hipMallocAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipFreeAsync = _libraries['libamdhip64.so'].hipFreeAsync
    hipFreeAsync.restype = hipError_t
    hipFreeAsync.argtypes = [ctypes.POINTER(None), hipStream_t]
except AttributeError:
    pass
try:
    hipMemPoolTrimTo = _libraries['libamdhip64.so'].hipMemPoolTrimTo
    hipMemPoolTrimTo.restype = hipError_t
    hipMemPoolTrimTo.argtypes = [hipMemPool_t, size_t]
except AttributeError:
    pass
try:
    hipMemPoolSetAttribute = _libraries['libamdhip64.so'].hipMemPoolSetAttribute
    hipMemPoolSetAttribute.restype = hipError_t
    hipMemPoolSetAttribute.argtypes = [hipMemPool_t, hipMemPoolAttr, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemPoolGetAttribute = _libraries['libamdhip64.so'].hipMemPoolGetAttribute
    hipMemPoolGetAttribute.restype = hipError_t
    hipMemPoolGetAttribute.argtypes = [hipMemPool_t, hipMemPoolAttr, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemPoolSetAccess = _libraries['libamdhip64.so'].hipMemPoolSetAccess
    hipMemPoolSetAccess.restype = hipError_t
    hipMemPoolSetAccess.argtypes = [hipMemPool_t, ctypes.POINTER(struct_hipMemAccessDesc), size_t]
except AttributeError:
    pass
try:
    hipMemPoolGetAccess = _libraries['libamdhip64.so'].hipMemPoolGetAccess
    hipMemPoolGetAccess.restype = hipError_t
    hipMemPoolGetAccess.argtypes = [ctypes.POINTER(hipMemAccessFlags), hipMemPool_t, ctypes.POINTER(struct_hipMemLocation)]
except AttributeError:
    pass
try:
    hipMemPoolCreate = _libraries['libamdhip64.so'].hipMemPoolCreate
    hipMemPoolCreate.restype = hipError_t
    hipMemPoolCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemPoolHandle_t)), ctypes.POINTER(struct_hipMemPoolProps)]
except AttributeError:
    pass
try:
    hipMemPoolDestroy = _libraries['libamdhip64.so'].hipMemPoolDestroy
    hipMemPoolDestroy.restype = hipError_t
    hipMemPoolDestroy.argtypes = [hipMemPool_t]
except AttributeError:
    pass
try:
    hipMallocFromPoolAsync = _libraries['libamdhip64.so'].hipMallocFromPoolAsync
    hipMallocFromPoolAsync.restype = hipError_t
    hipMallocFromPoolAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, hipMemPool_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemPoolExportToShareableHandle = _libraries['libamdhip64.so'].hipMemPoolExportToShareableHandle
    hipMemPoolExportToShareableHandle.restype = hipError_t
    hipMemPoolExportToShareableHandle.argtypes = [ctypes.POINTER(None), hipMemPool_t, hipMemAllocationHandleType, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMemPoolImportFromShareableHandle = _libraries['libamdhip64.so'].hipMemPoolImportFromShareableHandle
    hipMemPoolImportFromShareableHandle.restype = hipError_t
    hipMemPoolImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemPoolHandle_t)), ctypes.POINTER(None), hipMemAllocationHandleType, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMemPoolExportPointer = _libraries['libamdhip64.so'].hipMemPoolExportPointer
    hipMemPoolExportPointer.restype = hipError_t
    hipMemPoolExportPointer.argtypes = [ctypes.POINTER(struct_hipMemPoolPtrExportData), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemPoolImportPointer = _libraries['libamdhip64.so'].hipMemPoolImportPointer
    hipMemPoolImportPointer.restype = hipError_t
    hipMemPoolImportPointer.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), hipMemPool_t, ctypes.POINTER(struct_hipMemPoolPtrExportData)]
except AttributeError:
    pass
try:
    hipHostAlloc = _libraries['libamdhip64.so'].hipHostAlloc
    hipHostAlloc.restype = hipError_t
    hipHostAlloc.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipHostGetDevicePointer = _libraries['libamdhip64.so'].hipHostGetDevicePointer
    hipHostGetDevicePointer.restype = hipError_t
    hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipHostGetFlags = _libraries['libamdhip64.so'].hipHostGetFlags
    hipHostGetFlags.restype = hipError_t
    hipHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipHostRegister = _libraries['libamdhip64.so'].hipHostRegister
    hipHostRegister.restype = hipError_t
    hipHostRegister.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipHostUnregister = _libraries['libamdhip64.so'].hipHostUnregister
    hipHostUnregister.restype = hipError_t
    hipHostUnregister.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMallocPitch = _libraries['libamdhip64.so'].hipMallocPitch
    hipMallocPitch.restype = hipError_t
    hipMallocPitch.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), size_t, size_t]
except AttributeError:
    pass
try:
    hipMemAllocPitch = _libraries['libamdhip64.so'].hipMemAllocPitch
    hipMemAllocPitch.restype = hipError_t
    hipMemAllocPitch.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipFree = _libraries['libamdhip64.so'].hipFree
    hipFree.restype = hipError_t
    hipFree.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipFreeHost = _libraries['libamdhip64.so'].hipFreeHost
    hipFreeHost.restype = hipError_t
    hipFreeHost.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipHostFree = _libraries['libamdhip64.so'].hipHostFree
    hipHostFree.restype = hipError_t
    hipHostFree.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemcpy = _libraries['libamdhip64.so'].hipMemcpy
    hipMemcpy.restype = hipError_t
    hipMemcpy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpyWithStream = _libraries['libamdhip64.so'].hipMemcpyWithStream
    hipMemcpyWithStream.restype = hipError_t
    hipMemcpyWithStream.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyHtoD = _libraries['libamdhip64.so'].hipMemcpyHtoD
    hipMemcpyHtoD.restype = hipError_t
    hipMemcpyHtoD.argtypes = [hipDeviceptr_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipMemcpyDtoH = _libraries['libamdhip64.so'].hipMemcpyDtoH
    hipMemcpyDtoH.restype = hipError_t
    hipMemcpyDtoH.argtypes = [ctypes.POINTER(None), hipDeviceptr_t, size_t]
except AttributeError:
    pass
try:
    hipMemcpyDtoD = _libraries['libamdhip64.so'].hipMemcpyDtoD
    hipMemcpyDtoD.restype = hipError_t
    hipMemcpyDtoD.argtypes = [hipDeviceptr_t, hipDeviceptr_t, size_t]
except AttributeError:
    pass
try:
    hipMemcpyHtoDAsync = _libraries['libamdhip64.so'].hipMemcpyHtoDAsync
    hipMemcpyHtoDAsync.restype = hipError_t
    hipMemcpyHtoDAsync.argtypes = [hipDeviceptr_t, ctypes.POINTER(None), size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyDtoHAsync = _libraries['libamdhip64.so'].hipMemcpyDtoHAsync
    hipMemcpyDtoHAsync.restype = hipError_t
    hipMemcpyDtoHAsync.argtypes = [ctypes.POINTER(None), hipDeviceptr_t, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyDtoDAsync = _libraries['libamdhip64.so'].hipMemcpyDtoDAsync
    hipMemcpyDtoDAsync.restype = hipError_t
    hipMemcpyDtoDAsync.argtypes = [hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipModuleGetGlobal = _libraries['libamdhip64.so'].hipModuleGetGlobal
    hipModuleGetGlobal.restype = hipError_t
    hipModuleGetGlobal.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hipGetSymbolAddress = _libraries['libamdhip64.so'].hipGetSymbolAddress
    hipGetSymbolAddress.restype = hipError_t
    hipGetSymbolAddress.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipGetSymbolSize = _libraries['libamdhip64.so'].hipGetSymbolSize
    hipGetSymbolSize.restype = hipError_t
    hipGetSymbolSize.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemcpyToSymbol = _libraries['libamdhip64.so'].hipMemcpyToSymbol
    hipMemcpyToSymbol.restype = hipError_t
    hipMemcpyToSymbol.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpyToSymbolAsync = _libraries['libamdhip64.so'].hipMemcpyToSymbolAsync
    hipMemcpyToSymbolAsync.restype = hipError_t
    hipMemcpyToSymbolAsync.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyFromSymbol = _libraries['libamdhip64.so'].hipMemcpyFromSymbol
    hipMemcpyFromSymbol.restype = hipError_t
    hipMemcpyFromSymbol.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpyFromSymbolAsync = _libraries['libamdhip64.so'].hipMemcpyFromSymbolAsync
    hipMemcpyFromSymbolAsync.restype = hipError_t
    hipMemcpyFromSymbolAsync.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyAsync = _libraries['libamdhip64.so'].hipMemcpyAsync
    hipMemcpyAsync.restype = hipError_t
    hipMemcpyAsync.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemset = _libraries['libamdhip64.so'].hipMemset
    hipMemset.restype = hipError_t
    hipMemset.argtypes = [ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    hipMemsetD8 = _libraries['libamdhip64.so'].hipMemsetD8
    hipMemsetD8.restype = hipError_t
    hipMemsetD8.argtypes = [hipDeviceptr_t, ctypes.c_ubyte, size_t]
except AttributeError:
    pass
try:
    hipMemsetD8Async = _libraries['libamdhip64.so'].hipMemsetD8Async
    hipMemsetD8Async.restype = hipError_t
    hipMemsetD8Async.argtypes = [hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemsetD16 = _libraries['libamdhip64.so'].hipMemsetD16
    hipMemsetD16.restype = hipError_t
    hipMemsetD16.argtypes = [hipDeviceptr_t, ctypes.c_uint16, size_t]
except AttributeError:
    pass
try:
    hipMemsetD16Async = _libraries['libamdhip64.so'].hipMemsetD16Async
    hipMemsetD16Async.restype = hipError_t
    hipMemsetD16Async.argtypes = [hipDeviceptr_t, ctypes.c_uint16, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemsetD32 = _libraries['libamdhip64.so'].hipMemsetD32
    hipMemsetD32.restype = hipError_t
    hipMemsetD32.argtypes = [hipDeviceptr_t, ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    hipMemsetAsync = _libraries['libamdhip64.so'].hipMemsetAsync
    hipMemsetAsync.restype = hipError_t
    hipMemsetAsync.argtypes = [ctypes.POINTER(None), ctypes.c_int32, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemsetD32Async = _libraries['libamdhip64.so'].hipMemsetD32Async
    hipMemsetD32Async.restype = hipError_t
    hipMemsetD32Async.argtypes = [hipDeviceptr_t, ctypes.c_int32, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemset2D = _libraries['libamdhip64.so'].hipMemset2D
    hipMemset2D.restype = hipError_t
    hipMemset2D.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, size_t, size_t]
except AttributeError:
    pass
try:
    hipMemset2DAsync = _libraries['libamdhip64.so'].hipMemset2DAsync
    hipMemset2DAsync.restype = hipError_t
    hipMemset2DAsync.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, size_t, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipMemset3D = _libraries['libamdhip64.so'].hipMemset3D
    hipMemset3D.restype = hipError_t
    hipMemset3D.argtypes = [hipPitchedPtr, ctypes.c_int32, hipExtent]
except AttributeError:
    pass
try:
    hipMemset3DAsync = _libraries['libamdhip64.so'].hipMemset3DAsync
    hipMemset3DAsync.restype = hipError_t
    hipMemset3DAsync.argtypes = [hipPitchedPtr, ctypes.c_int32, hipExtent, hipStream_t]
except AttributeError:
    pass
try:
    hipMemGetInfo = _libraries['libamdhip64.so'].hipMemGetInfo
    hipMemGetInfo.restype = hipError_t
    hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipMemPtrGetInfo = _libraries['libamdhip64.so'].hipMemPtrGetInfo
    hipMemPtrGetInfo.restype = hipError_t
    hipMemPtrGetInfo.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipMallocArray = _libraries['libamdhip64.so'].hipMallocArray
    hipMallocArray.restype = hipError_t
    hipMallocArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), ctypes.POINTER(struct_hipChannelFormatDesc), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipArrayCreate = _libraries['libamdhip64.so'].hipArrayCreate
    hipArrayCreate.restype = hipError_t
    hipArrayCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), ctypes.POINTER(struct_HIP_ARRAY_DESCRIPTOR)]
except AttributeError:
    pass
try:
    hipArrayDestroy = _libraries['libamdhip64.so'].hipArrayDestroy
    hipArrayDestroy.restype = hipError_t
    hipArrayDestroy.argtypes = [hipArray_t]
except AttributeError:
    pass
try:
    hipArray3DCreate = _libraries['libamdhip64.so'].hipArray3DCreate
    hipArray3DCreate.restype = hipError_t
    hipArray3DCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), ctypes.POINTER(struct_HIP_ARRAY3D_DESCRIPTOR)]
except AttributeError:
    pass
try:
    hipMalloc3D = _libraries['libamdhip64.so'].hipMalloc3D
    hipMalloc3D.restype = hipError_t
    hipMalloc3D.argtypes = [ctypes.POINTER(struct_hipPitchedPtr), hipExtent]
except AttributeError:
    pass
try:
    hipFreeArray = _libraries['libamdhip64.so'].hipFreeArray
    hipFreeArray.restype = hipError_t
    hipFreeArray.argtypes = [hipArray_t]
except AttributeError:
    pass
try:
    hipMalloc3DArray = _libraries['libamdhip64.so'].hipMalloc3DArray
    hipMalloc3DArray.restype = hipError_t
    hipMalloc3DArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), ctypes.POINTER(struct_hipChannelFormatDesc), struct_hipExtent, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipArrayGetInfo = _libraries['libamdhip64.so'].hipArrayGetInfo
    hipArrayGetInfo.restype = hipError_t
    hipArrayGetInfo.argtypes = [ctypes.POINTER(struct_hipChannelFormatDesc), ctypes.POINTER(struct_hipExtent), ctypes.POINTER(ctypes.c_uint32), hipArray_t]
except AttributeError:
    pass
try:
    hipArrayGetDescriptor = _libraries['libamdhip64.so'].hipArrayGetDescriptor
    hipArrayGetDescriptor.restype = hipError_t
    hipArrayGetDescriptor.argtypes = [ctypes.POINTER(struct_HIP_ARRAY_DESCRIPTOR), hipArray_t]
except AttributeError:
    pass
try:
    hipArray3DGetDescriptor = _libraries['libamdhip64.so'].hipArray3DGetDescriptor
    hipArray3DGetDescriptor.restype = hipError_t
    hipArray3DGetDescriptor.argtypes = [ctypes.POINTER(struct_HIP_ARRAY3D_DESCRIPTOR), hipArray_t]
except AttributeError:
    pass
try:
    hipMemcpy2D = _libraries['libamdhip64.so'].hipMemcpy2D
    hipMemcpy2D.restype = hipError_t
    hipMemcpy2D.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(None), size_t, size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpyParam2D = _libraries['libamdhip64.so'].hipMemcpyParam2D
    hipMemcpyParam2D.restype = hipError_t
    hipMemcpyParam2D.argtypes = [ctypes.POINTER(struct_hip_Memcpy2D)]
except AttributeError:
    pass
try:
    hipMemcpyParam2DAsync = _libraries['libamdhip64.so'].hipMemcpyParam2DAsync
    hipMemcpyParam2DAsync.restype = hipError_t
    hipMemcpyParam2DAsync.argtypes = [ctypes.POINTER(struct_hip_Memcpy2D), hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpy2DAsync = _libraries['libamdhip64.so'].hipMemcpy2DAsync
    hipMemcpy2DAsync.restype = hipError_t
    hipMemcpy2DAsync.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(None), size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpy2DToArray = _libraries['libamdhip64.so'].hipMemcpy2DToArray
    hipMemcpy2DToArray.restype = hipError_t
    hipMemcpy2DToArray.argtypes = [hipArray_t, size_t, size_t, ctypes.POINTER(None), size_t, size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpy2DToArrayAsync = _libraries['libamdhip64.so'].hipMemcpy2DToArrayAsync
    hipMemcpy2DToArrayAsync.restype = hipError_t
    hipMemcpy2DToArrayAsync.argtypes = [hipArray_t, size_t, size_t, ctypes.POINTER(None), size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyToArray = _libraries['libamdhip64.so'].hipMemcpyToArray
    hipMemcpyToArray.restype = hipError_t
    hipMemcpyToArray.argtypes = [hipArray_t, size_t, size_t, ctypes.POINTER(None), size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpyFromArray = _libraries['libamdhip64.so'].hipMemcpyFromArray
    hipMemcpyFromArray.restype = hipError_t
    hipMemcpyFromArray.argtypes = [ctypes.POINTER(None), hipArray_const_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpy2DFromArray = _libraries['libamdhip64.so'].hipMemcpy2DFromArray
    hipMemcpy2DFromArray.restype = hipError_t
    hipMemcpy2DFromArray.argtypes = [ctypes.POINTER(None), size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipMemcpy2DFromArrayAsync = _libraries['libamdhip64.so'].hipMemcpy2DFromArrayAsync
    hipMemcpy2DFromArrayAsync.restype = hipError_t
    hipMemcpy2DFromArrayAsync.argtypes = [ctypes.POINTER(None), size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError:
    pass
try:
    hipMemcpyAtoH = _libraries['libamdhip64.so'].hipMemcpyAtoH
    hipMemcpyAtoH.restype = hipError_t
    hipMemcpyAtoH.argtypes = [ctypes.POINTER(None), hipArray_t, size_t, size_t]
except AttributeError:
    pass
try:
    hipMemcpyHtoA = _libraries['libamdhip64.so'].hipMemcpyHtoA
    hipMemcpyHtoA.restype = hipError_t
    hipMemcpyHtoA.argtypes = [hipArray_t, size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipMemcpy3D = _libraries['libamdhip64.so'].hipMemcpy3D
    hipMemcpy3D.restype = hipError_t
    hipMemcpy3D.argtypes = [ctypes.POINTER(struct_hipMemcpy3DParms)]
except AttributeError:
    pass
try:
    hipMemcpy3DAsync = _libraries['libamdhip64.so'].hipMemcpy3DAsync
    hipMemcpy3DAsync.restype = hipError_t
    hipMemcpy3DAsync.argtypes = [ctypes.POINTER(struct_hipMemcpy3DParms), hipStream_t]
except AttributeError:
    pass
try:
    hipDrvMemcpy3D = _libraries['libamdhip64.so'].hipDrvMemcpy3D
    hipDrvMemcpy3D.restype = hipError_t
    hipDrvMemcpy3D.argtypes = [ctypes.POINTER(struct_HIP_MEMCPY3D)]
except AttributeError:
    pass
try:
    hipDrvMemcpy3DAsync = _libraries['libamdhip64.so'].hipDrvMemcpy3DAsync
    hipDrvMemcpy3DAsync.restype = hipError_t
    hipDrvMemcpy3DAsync.argtypes = [ctypes.POINTER(struct_HIP_MEMCPY3D), hipStream_t]
except AttributeError:
    pass
try:
    hipDeviceCanAccessPeer = _libraries['libamdhip64.so'].hipDeviceCanAccessPeer
    hipDeviceCanAccessPeer.restype = hipError_t
    hipDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipDeviceEnablePeerAccess = _libraries['libamdhip64.so'].hipDeviceEnablePeerAccess
    hipDeviceEnablePeerAccess.restype = hipError_t
    hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipDeviceDisablePeerAccess = _libraries['libamdhip64.so'].hipDeviceDisablePeerAccess
    hipDeviceDisablePeerAccess.restype = hipError_t
    hipDeviceDisablePeerAccess.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    hipMemGetAddressRange = _libraries['libamdhip64.so'].hipMemGetAddressRange
    hipMemGetAddressRange.restype = hipError_t
    hipMemGetAddressRange.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), hipDeviceptr_t]
except AttributeError:
    pass
try:
    hipMemcpyPeer = _libraries['libamdhip64.so'].hipMemcpyPeer
    hipMemcpyPeer.restype = hipError_t
    hipMemcpyPeer.argtypes = [ctypes.POINTER(None), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    hipMemcpyPeerAsync = _libraries['libamdhip64.so'].hipMemcpyPeerAsync
    hipMemcpyPeerAsync.restype = hipError_t
    hipMemcpyPeerAsync.argtypes = [ctypes.POINTER(None), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipCtxCreate = _libraries['libamdhip64.so'].hipCtxCreate
    hipCtxCreate.restype = hipError_t
    hipCtxCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipCtx_t)), ctypes.c_uint32, hipDevice_t]
except AttributeError:
    pass
try:
    hipCtxDestroy = _libraries['libamdhip64.so'].hipCtxDestroy
    hipCtxDestroy.restype = hipError_t
    hipCtxDestroy.argtypes = [hipCtx_t]
except AttributeError:
    pass
try:
    hipCtxPopCurrent = _libraries['libamdhip64.so'].hipCtxPopCurrent
    hipCtxPopCurrent.restype = hipError_t
    hipCtxPopCurrent.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipCtx_t))]
except AttributeError:
    pass
try:
    hipCtxPushCurrent = _libraries['libamdhip64.so'].hipCtxPushCurrent
    hipCtxPushCurrent.restype = hipError_t
    hipCtxPushCurrent.argtypes = [hipCtx_t]
except AttributeError:
    pass
try:
    hipCtxSetCurrent = _libraries['libamdhip64.so'].hipCtxSetCurrent
    hipCtxSetCurrent.restype = hipError_t
    hipCtxSetCurrent.argtypes = [hipCtx_t]
except AttributeError:
    pass
try:
    hipCtxGetCurrent = _libraries['libamdhip64.so'].hipCtxGetCurrent
    hipCtxGetCurrent.restype = hipError_t
    hipCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipCtx_t))]
except AttributeError:
    pass
try:
    hipCtxGetDevice = _libraries['libamdhip64.so'].hipCtxGetDevice
    hipCtxGetDevice.restype = hipError_t
    hipCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipCtxGetApiVersion = _libraries['libamdhip64.so'].hipCtxGetApiVersion
    hipCtxGetApiVersion.restype = hipError_t
    hipCtxGetApiVersion.argtypes = [hipCtx_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipCtxGetCacheConfig = _libraries['libamdhip64.so'].hipCtxGetCacheConfig
    hipCtxGetCacheConfig.restype = hipError_t
    hipCtxGetCacheConfig.argtypes = [ctypes.POINTER(hipFuncCache_t)]
except AttributeError:
    pass
try:
    hipCtxSetCacheConfig = _libraries['libamdhip64.so'].hipCtxSetCacheConfig
    hipCtxSetCacheConfig.restype = hipError_t
    hipCtxSetCacheConfig.argtypes = [hipFuncCache_t]
except AttributeError:
    pass
try:
    hipCtxSetSharedMemConfig = _libraries['libamdhip64.so'].hipCtxSetSharedMemConfig
    hipCtxSetSharedMemConfig.restype = hipError_t
    hipCtxSetSharedMemConfig.argtypes = [hipSharedMemConfig]
except AttributeError:
    pass
try:
    hipCtxGetSharedMemConfig = _libraries['libamdhip64.so'].hipCtxGetSharedMemConfig
    hipCtxGetSharedMemConfig.restype = hipError_t
    hipCtxGetSharedMemConfig.argtypes = [ctypes.POINTER(hipSharedMemConfig)]
except AttributeError:
    pass
try:
    hipCtxSynchronize = _libraries['libamdhip64.so'].hipCtxSynchronize
    hipCtxSynchronize.restype = hipError_t
    hipCtxSynchronize.argtypes = []
except AttributeError:
    pass
try:
    hipCtxGetFlags = _libraries['libamdhip64.so'].hipCtxGetFlags
    hipCtxGetFlags.restype = hipError_t
    hipCtxGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipCtxEnablePeerAccess = _libraries['libamdhip64.so'].hipCtxEnablePeerAccess
    hipCtxEnablePeerAccess.restype = hipError_t
    hipCtxEnablePeerAccess.argtypes = [hipCtx_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipCtxDisablePeerAccess = _libraries['libamdhip64.so'].hipCtxDisablePeerAccess
    hipCtxDisablePeerAccess.restype = hipError_t
    hipCtxDisablePeerAccess.argtypes = [hipCtx_t]
except AttributeError:
    pass
try:
    hipDevicePrimaryCtxGetState = _libraries['libamdhip64.so'].hipDevicePrimaryCtxGetState
    hipDevicePrimaryCtxGetState.restype = hipError_t
    hipDevicePrimaryCtxGetState.argtypes = [hipDevice_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hipDevicePrimaryCtxRelease = _libraries['libamdhip64.so'].hipDevicePrimaryCtxRelease
    hipDevicePrimaryCtxRelease.restype = hipError_t
    hipDevicePrimaryCtxRelease.argtypes = [hipDevice_t]
except AttributeError:
    pass
try:
    hipDevicePrimaryCtxRetain = _libraries['libamdhip64.so'].hipDevicePrimaryCtxRetain
    hipDevicePrimaryCtxRetain.restype = hipError_t
    hipDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipCtx_t)), hipDevice_t]
except AttributeError:
    pass
try:
    hipDevicePrimaryCtxReset = _libraries['libamdhip64.so'].hipDevicePrimaryCtxReset
    hipDevicePrimaryCtxReset.restype = hipError_t
    hipDevicePrimaryCtxReset.argtypes = [hipDevice_t]
except AttributeError:
    pass
try:
    hipDevicePrimaryCtxSetFlags = _libraries['libamdhip64.so'].hipDevicePrimaryCtxSetFlags
    hipDevicePrimaryCtxSetFlags.restype = hipError_t
    hipDevicePrimaryCtxSetFlags.argtypes = [hipDevice_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipModuleLoad = _libraries['libamdhip64.so'].hipModuleLoad
    hipModuleLoad.restype = hipError_t
    hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipModule_t)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hipModuleUnload = _libraries['libamdhip64.so'].hipModuleUnload
    hipModuleUnload.restype = hipError_t
    hipModuleUnload.argtypes = [hipModule_t]
except AttributeError:
    pass
try:
    hipModuleGetFunction = _libraries['libamdhip64.so'].hipModuleGetFunction
    hipModuleGetFunction.restype = hipError_t
    hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipModuleSymbol_t)), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hipFuncGetAttributes = _libraries['libamdhip64.so'].hipFuncGetAttributes
    hipFuncGetAttributes.restype = hipError_t
    hipFuncGetAttributes.argtypes = [ctypes.POINTER(struct_hipFuncAttributes), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipFuncGetAttribute = _libraries['libamdhip64.so'].hipFuncGetAttribute
    hipFuncGetAttribute.restype = hipError_t
    hipFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), hipFunction_attribute, hipFunction_t]
except AttributeError:
    pass
class struct_textureReference(Structure):
    pass

class struct___hip_texture(Structure):
    pass


# values for enumeration 'hipTextureReadMode'
hipTextureReadMode__enumvalues = {
    0: 'hipReadModeElementType',
    1: 'hipReadModeNormalizedFloat',
}
hipReadModeElementType = 0
hipReadModeNormalizedFloat = 1
hipTextureReadMode = ctypes.c_uint32 # enum

# values for enumeration 'hipTextureFilterMode'
hipTextureFilterMode__enumvalues = {
    0: 'hipFilterModePoint',
    1: 'hipFilterModeLinear',
}
hipFilterModePoint = 0
hipFilterModeLinear = 1
hipTextureFilterMode = ctypes.c_uint32 # enum

# values for enumeration 'hipTextureAddressMode'
hipTextureAddressMode__enumvalues = {
    0: 'hipAddressModeWrap',
    1: 'hipAddressModeClamp',
    2: 'hipAddressModeMirror',
    3: 'hipAddressModeBorder',
}
hipAddressModeWrap = 0
hipAddressModeClamp = 1
hipAddressModeMirror = 2
hipAddressModeBorder = 3
hipTextureAddressMode = ctypes.c_uint32 # enum
struct_textureReference._pack_ = 1 # source:False
struct_textureReference._fields_ = [
    ('normalized', ctypes.c_int32),
    ('readMode', hipTextureReadMode),
    ('filterMode', hipTextureFilterMode),
    ('addressMode', hipTextureAddressMode * 3),
    ('channelDesc', struct_hipChannelFormatDesc),
    ('sRGB', ctypes.c_int32),
    ('maxAnisotropy', ctypes.c_uint32),
    ('mipmapFilterMode', hipTextureFilterMode),
    ('mipmapLevelBias', ctypes.c_float),
    ('minMipmapLevelClamp', ctypes.c_float),
    ('maxMipmapLevelClamp', ctypes.c_float),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('textureObject', ctypes.POINTER(struct___hip_texture)),
    ('numChannels', ctypes.c_int32),
    ('format', hipArray_Format),
]

try:
    hipModuleGetTexRef = _libraries['libamdhip64.so'].hipModuleGetTexRef
    hipModuleGetTexRef.restype = hipError_t
    hipModuleGetTexRef.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_textureReference)), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hipModuleLoadData = _libraries['libamdhip64.so'].hipModuleLoadData
    hipModuleLoadData.restype = hipError_t
    hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipModule_t)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipModuleLoadDataEx = _libraries['libamdhip64.so'].hipModuleLoadDataEx
    hipModuleLoadDataEx.restype = hipError_t
    hipModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipModule_t)), ctypes.POINTER(None), ctypes.c_uint32, ctypes.POINTER(hipJitOption), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hipModuleLaunchKernel = _libraries['libamdhip64.so'].hipModuleLaunchKernel
    hipModuleLaunchKernel.restype = hipError_t
    hipModuleLaunchKernel.argtypes = [hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hipModuleLaunchCooperativeKernel = _libraries['libamdhip64.so'].hipModuleLaunchCooperativeKernel
    hipModuleLaunchCooperativeKernel.restype = hipError_t
    hipModuleLaunchCooperativeKernel.argtypes = [hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hipModuleLaunchCooperativeKernelMultiDevice = _libraries['libamdhip64.so'].hipModuleLaunchCooperativeKernelMultiDevice
    hipModuleLaunchCooperativeKernelMultiDevice.restype = hipError_t
    hipModuleLaunchCooperativeKernelMultiDevice.argtypes = [ctypes.POINTER(struct_hipFunctionLaunchParams_t), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipLaunchCooperativeKernel = _libraries['libamdhip64.so'].hipLaunchCooperativeKernel
    hipLaunchCooperativeKernel.restype = hipError_t
    hipLaunchCooperativeKernel.argtypes = [ctypes.POINTER(None), dim3, dim3, ctypes.POINTER(ctypes.POINTER(None)), ctypes.c_uint32, hipStream_t]
except AttributeError:
    pass
try:
    hipLaunchCooperativeKernelMultiDevice = _libraries['libamdhip64.so'].hipLaunchCooperativeKernelMultiDevice
    hipLaunchCooperativeKernelMultiDevice.restype = hipError_t
    hipLaunchCooperativeKernelMultiDevice.argtypes = [ctypes.POINTER(struct_hipLaunchParams_t), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipExtLaunchMultiKernelMultiDevice = _libraries['libamdhip64.so'].hipExtLaunchMultiKernelMultiDevice
    hipExtLaunchMultiKernelMultiDevice.restype = hipError_t
    hipExtLaunchMultiKernelMultiDevice.argtypes = [ctypes.POINTER(struct_hipLaunchParams_t), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipModuleOccupancyMaxPotentialBlockSize = _libraries['libamdhip64.so'].hipModuleOccupancyMaxPotentialBlockSize
    hipModuleOccupancyMaxPotentialBlockSize.restype = hipError_t
    hipModuleOccupancyMaxPotentialBlockSize.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipModuleOccupancyMaxPotentialBlockSizeWithFlags = _libraries['libamdhip64.so'].hipModuleOccupancyMaxPotentialBlockSizeWithFlags
    hipModuleOccupancyMaxPotentialBlockSizeWithFlags.restype = hipError_t
    hipModuleOccupancyMaxPotentialBlockSizeWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor = _libraries['libamdhip64.so'].hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.restype = hipError_t
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = _libraries['libamdhip64.so'].hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.restype = hipError_t
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipOccupancyMaxActiveBlocksPerMultiprocessor = _libraries['libamdhip64.so'].hipOccupancyMaxActiveBlocksPerMultiprocessor
    hipOccupancyMaxActiveBlocksPerMultiprocessor.restype = hipError_t
    hipOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = _libraries['libamdhip64.so'].hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.restype = hipError_t
    hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(None), ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipOccupancyMaxPotentialBlockSize = _libraries['libamdhip64.so'].hipOccupancyMaxPotentialBlockSize
    hipOccupancyMaxPotentialBlockSize.restype = hipError_t
    hipOccupancyMaxPotentialBlockSize.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipProfilerStart = _libraries['libamdhip64.so'].hipProfilerStart
    hipProfilerStart.restype = hipError_t
    hipProfilerStart.argtypes = []
except AttributeError:
    pass
try:
    hipProfilerStop = _libraries['libamdhip64.so'].hipProfilerStop
    hipProfilerStop.restype = hipError_t
    hipProfilerStop.argtypes = []
except AttributeError:
    pass
try:
    hipConfigureCall = _libraries['libamdhip64.so'].hipConfigureCall
    hipConfigureCall.restype = hipError_t
    hipConfigureCall.argtypes = [dim3, dim3, size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipSetupArgument = _libraries['libamdhip64.so'].hipSetupArgument
    hipSetupArgument.restype = hipError_t
    hipSetupArgument.argtypes = [ctypes.POINTER(None), size_t, size_t]
except AttributeError:
    pass
try:
    hipLaunchByPtr = _libraries['libamdhip64.so'].hipLaunchByPtr
    hipLaunchByPtr.restype = hipError_t
    hipLaunchByPtr.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    __hipPushCallConfiguration = _libraries['libamdhip64.so'].__hipPushCallConfiguration
    __hipPushCallConfiguration.restype = hipError_t
    __hipPushCallConfiguration.argtypes = [dim3, dim3, size_t, hipStream_t]
except AttributeError:
    pass
try:
    __hipPopCallConfiguration = _libraries['libamdhip64.so'].__hipPopCallConfiguration
    __hipPopCallConfiguration.restype = hipError_t
    __hipPopCallConfiguration.argtypes = [ctypes.POINTER(struct_dim3), ctypes.POINTER(struct_dim3), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(struct_ihipStream_t))]
except AttributeError:
    pass
try:
    hipLaunchKernel = _libraries['libamdhip64.so'].hipLaunchKernel
    hipLaunchKernel.restype = hipError_t
    hipLaunchKernel.argtypes = [ctypes.POINTER(None), dim3, dim3, ctypes.POINTER(ctypes.POINTER(None)), size_t, hipStream_t]
except AttributeError:
    pass
try:
    hipLaunchHostFunc = _libraries['libamdhip64.so'].hipLaunchHostFunc
    hipLaunchHostFunc.restype = hipError_t
    hipLaunchHostFunc.argtypes = [hipStream_t, hipHostFn_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipDrvMemcpy2DUnaligned = _libraries['libamdhip64.so'].hipDrvMemcpy2DUnaligned
    hipDrvMemcpy2DUnaligned.restype = hipError_t
    hipDrvMemcpy2DUnaligned.argtypes = [ctypes.POINTER(struct_hip_Memcpy2D)]
except AttributeError:
    pass
try:
    hipExtLaunchKernel = _libraries['libamdhip64.so'].hipExtLaunchKernel
    hipExtLaunchKernel.restype = hipError_t
    hipExtLaunchKernel.argtypes = [ctypes.POINTER(None), dim3, dim3, ctypes.POINTER(ctypes.POINTER(None)), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32]
except AttributeError:
    pass
class struct_hipTextureDesc(Structure):
    pass

struct_hipTextureDesc._pack_ = 1 # source:False
struct_hipTextureDesc._fields_ = [
    ('addressMode', hipTextureAddressMode * 3),
    ('filterMode', hipTextureFilterMode),
    ('readMode', hipTextureReadMode),
    ('sRGB', ctypes.c_int32),
    ('borderColor', ctypes.c_float * 4),
    ('normalizedCoords', ctypes.c_int32),
    ('maxAnisotropy', ctypes.c_uint32),
    ('mipmapFilterMode', hipTextureFilterMode),
    ('mipmapLevelBias', ctypes.c_float),
    ('minMipmapLevelClamp', ctypes.c_float),
    ('maxMipmapLevelClamp', ctypes.c_float),
]

try:
    hipCreateTextureObject = _libraries['libamdhip64.so'].hipCreateTextureObject
    hipCreateTextureObject.restype = hipError_t
    hipCreateTextureObject.argtypes = [ctypes.POINTER(ctypes.POINTER(struct___hip_texture)), ctypes.POINTER(struct_hipResourceDesc), ctypes.POINTER(struct_hipTextureDesc), ctypes.POINTER(struct_hipResourceViewDesc)]
except AttributeError:
    pass
hipTextureObject_t = ctypes.POINTER(struct___hip_texture)
try:
    hipDestroyTextureObject = _libraries['libamdhip64.so'].hipDestroyTextureObject
    hipDestroyTextureObject.restype = hipError_t
    hipDestroyTextureObject.argtypes = [hipTextureObject_t]
except AttributeError:
    pass
try:
    hipGetChannelDesc = _libraries['libamdhip64.so'].hipGetChannelDesc
    hipGetChannelDesc.restype = hipError_t
    hipGetChannelDesc.argtypes = [ctypes.POINTER(struct_hipChannelFormatDesc), hipArray_const_t]
except AttributeError:
    pass
try:
    hipGetTextureObjectResourceDesc = _libraries['libamdhip64.so'].hipGetTextureObjectResourceDesc
    hipGetTextureObjectResourceDesc.restype = hipError_t
    hipGetTextureObjectResourceDesc.argtypes = [ctypes.POINTER(struct_hipResourceDesc), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipGetTextureObjectResourceViewDesc = _libraries['libamdhip64.so'].hipGetTextureObjectResourceViewDesc
    hipGetTextureObjectResourceViewDesc.restype = hipError_t
    hipGetTextureObjectResourceViewDesc.argtypes = [ctypes.POINTER(struct_hipResourceViewDesc), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipGetTextureObjectTextureDesc = _libraries['libamdhip64.so'].hipGetTextureObjectTextureDesc
    hipGetTextureObjectTextureDesc.restype = hipError_t
    hipGetTextureObjectTextureDesc.argtypes = [ctypes.POINTER(struct_hipTextureDesc), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipTexObjectCreate = _libraries['libamdhip64.so'].hipTexObjectCreate
    hipTexObjectCreate.restype = hipError_t
    hipTexObjectCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct___hip_texture)), ctypes.POINTER(struct_HIP_RESOURCE_DESC_st), ctypes.POINTER(struct_HIP_TEXTURE_DESC_st), ctypes.POINTER(struct_HIP_RESOURCE_VIEW_DESC_st)]
except AttributeError:
    pass
try:
    hipTexObjectDestroy = _libraries['libamdhip64.so'].hipTexObjectDestroy
    hipTexObjectDestroy.restype = hipError_t
    hipTexObjectDestroy.argtypes = [hipTextureObject_t]
except AttributeError:
    pass
try:
    hipTexObjectGetResourceDesc = _libraries['libamdhip64.so'].hipTexObjectGetResourceDesc
    hipTexObjectGetResourceDesc.restype = hipError_t
    hipTexObjectGetResourceDesc.argtypes = [ctypes.POINTER(struct_HIP_RESOURCE_DESC_st), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipTexObjectGetResourceViewDesc = _libraries['libamdhip64.so'].hipTexObjectGetResourceViewDesc
    hipTexObjectGetResourceViewDesc.restype = hipError_t
    hipTexObjectGetResourceViewDesc.argtypes = [ctypes.POINTER(struct_HIP_RESOURCE_VIEW_DESC_st), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipTexObjectGetTextureDesc = _libraries['libamdhip64.so'].hipTexObjectGetTextureDesc
    hipTexObjectGetTextureDesc.restype = hipError_t
    hipTexObjectGetTextureDesc.argtypes = [ctypes.POINTER(struct_HIP_TEXTURE_DESC_st), hipTextureObject_t]
except AttributeError:
    pass
try:
    hipMallocMipmappedArray = _libraries['libamdhip64.so'].hipMallocMipmappedArray
    hipMallocMipmappedArray.restype = hipError_t
    hipMallocMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipMipmappedArray)), ctypes.POINTER(struct_hipChannelFormatDesc), struct_hipExtent, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipFreeMipmappedArray = _libraries['libamdhip64.so'].hipFreeMipmappedArray
    hipFreeMipmappedArray.restype = hipError_t
    hipFreeMipmappedArray.argtypes = [hipMipmappedArray_t]
except AttributeError:
    pass
try:
    hipGetMipmappedArrayLevel = _libraries['libamdhip64.so'].hipGetMipmappedArrayLevel
    hipGetMipmappedArrayLevel.restype = hipError_t
    hipGetMipmappedArrayLevel.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), hipMipmappedArray_const_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMipmappedArrayCreate = _libraries['libamdhip64.so'].hipMipmappedArrayCreate
    hipMipmappedArrayCreate.restype = hipError_t
    hipMipmappedArrayCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipMipmappedArray)), ctypes.POINTER(struct_HIP_ARRAY3D_DESCRIPTOR), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipMipmappedArrayDestroy = _libraries['libamdhip64.so'].hipMipmappedArrayDestroy
    hipMipmappedArrayDestroy.restype = hipError_t
    hipMipmappedArrayDestroy.argtypes = [hipMipmappedArray_t]
except AttributeError:
    pass
try:
    hipMipmappedArrayGetLevel = _libraries['libamdhip64.so'].hipMipmappedArrayGetLevel
    hipMipmappedArrayGetLevel.restype = hipError_t
    hipMipmappedArrayGetLevel.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), hipMipmappedArray_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipBindTextureToMipmappedArray = _libraries['libamdhip64.so'].hipBindTextureToMipmappedArray
    hipBindTextureToMipmappedArray.restype = hipError_t
    hipBindTextureToMipmappedArray.argtypes = [ctypes.POINTER(struct_textureReference), hipMipmappedArray_const_t, ctypes.POINTER(struct_hipChannelFormatDesc)]
except AttributeError:
    pass
try:
    hipGetTextureReference = _libraries['libamdhip64.so'].hipGetTextureReference
    hipGetTextureReference.restype = hipError_t
    hipGetTextureReference.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_textureReference)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipTexRefSetAddressMode = _libraries['libamdhip64.so'].hipTexRefSetAddressMode
    hipTexRefSetAddressMode.restype = hipError_t
    hipTexRefSetAddressMode.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.c_int32, hipTextureAddressMode]
except AttributeError:
    pass
try:
    hipTexRefSetArray = _libraries['libamdhip64.so'].hipTexRefSetArray
    hipTexRefSetArray.restype = hipError_t
    hipTexRefSetArray.argtypes = [ctypes.POINTER(struct_textureReference), hipArray_const_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipTexRefSetFilterMode = _libraries['libamdhip64.so'].hipTexRefSetFilterMode
    hipTexRefSetFilterMode.restype = hipError_t
    hipTexRefSetFilterMode.argtypes = [ctypes.POINTER(struct_textureReference), hipTextureFilterMode]
except AttributeError:
    pass
try:
    hipTexRefSetFlags = _libraries['libamdhip64.so'].hipTexRefSetFlags
    hipTexRefSetFlags.restype = hipError_t
    hipTexRefSetFlags.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipTexRefSetFormat = _libraries['libamdhip64.so'].hipTexRefSetFormat
    hipTexRefSetFormat.restype = hipError_t
    hipTexRefSetFormat.argtypes = [ctypes.POINTER(struct_textureReference), hipArray_Format, ctypes.c_int32]
except AttributeError:
    pass
try:
    hipBindTexture = _libraries['libamdhip64.so'].hipBindTexture
    hipBindTexture.restype = hipError_t
    hipBindTexture.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_textureReference), ctypes.POINTER(None), ctypes.POINTER(struct_hipChannelFormatDesc), size_t]
except AttributeError:
    pass
try:
    hipBindTexture2D = _libraries['libamdhip64.so'].hipBindTexture2D
    hipBindTexture2D.restype = hipError_t
    hipBindTexture2D.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_textureReference), ctypes.POINTER(None), ctypes.POINTER(struct_hipChannelFormatDesc), size_t, size_t, size_t]
except AttributeError:
    pass
try:
    hipBindTextureToArray = _libraries['libamdhip64.so'].hipBindTextureToArray
    hipBindTextureToArray.restype = hipError_t
    hipBindTextureToArray.argtypes = [ctypes.POINTER(struct_textureReference), hipArray_const_t, ctypes.POINTER(struct_hipChannelFormatDesc)]
except AttributeError:
    pass
try:
    hipGetTextureAlignmentOffset = _libraries['libamdhip64.so'].hipGetTextureAlignmentOffset
    hipGetTextureAlignmentOffset.restype = hipError_t
    hipGetTextureAlignmentOffset.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipUnbindTexture = _libraries['libamdhip64.so'].hipUnbindTexture
    hipUnbindTexture.restype = hipError_t
    hipUnbindTexture.argtypes = [ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetAddress = _libraries['libamdhip64.so'].hipTexRefGetAddress
    hipTexRefGetAddress.restype = hipError_t
    hipTexRefGetAddress.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetAddressMode = _libraries['libamdhip64.so'].hipTexRefGetAddressMode
    hipTexRefGetAddressMode.restype = hipError_t
    hipTexRefGetAddressMode.argtypes = [ctypes.POINTER(hipTextureAddressMode), ctypes.POINTER(struct_textureReference), ctypes.c_int32]
except AttributeError:
    pass
try:
    hipTexRefGetFilterMode = _libraries['libamdhip64.so'].hipTexRefGetFilterMode
    hipTexRefGetFilterMode.restype = hipError_t
    hipTexRefGetFilterMode.argtypes = [ctypes.POINTER(hipTextureFilterMode), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetFlags = _libraries['libamdhip64.so'].hipTexRefGetFlags
    hipTexRefGetFlags.restype = hipError_t
    hipTexRefGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetFormat = _libraries['libamdhip64.so'].hipTexRefGetFormat
    hipTexRefGetFormat.restype = hipError_t
    hipTexRefGetFormat.argtypes = [ctypes.POINTER(hipArray_Format), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetMaxAnisotropy = _libraries['libamdhip64.so'].hipTexRefGetMaxAnisotropy
    hipTexRefGetMaxAnisotropy.restype = hipError_t
    hipTexRefGetMaxAnisotropy.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetMipmapFilterMode = _libraries['libamdhip64.so'].hipTexRefGetMipmapFilterMode
    hipTexRefGetMipmapFilterMode.restype = hipError_t
    hipTexRefGetMipmapFilterMode.argtypes = [ctypes.POINTER(hipTextureFilterMode), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetMipmapLevelBias = _libraries['libamdhip64.so'].hipTexRefGetMipmapLevelBias
    hipTexRefGetMipmapLevelBias.restype = hipError_t
    hipTexRefGetMipmapLevelBias.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetMipmapLevelClamp = _libraries['libamdhip64.so'].hipTexRefGetMipmapLevelClamp
    hipTexRefGetMipmapLevelClamp.restype = hipError_t
    hipTexRefGetMipmapLevelClamp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefGetMipMappedArray = _libraries['FIXME_STUB'].hipTexRefGetMipMappedArray
    hipTexRefGetMipMappedArray.restype = hipError_t
    hipTexRefGetMipMappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipMipmappedArray)), ctypes.POINTER(struct_textureReference)]
except AttributeError:
    pass
try:
    hipTexRefSetAddress = _libraries['libamdhip64.so'].hipTexRefSetAddress
    hipTexRefSetAddress.restype = hipError_t
    hipTexRefSetAddress.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_textureReference), hipDeviceptr_t, size_t]
except AttributeError:
    pass
try:
    hipTexRefSetAddress2D = _libraries['libamdhip64.so'].hipTexRefSetAddress2D
    hipTexRefSetAddress2D.restype = hipError_t
    hipTexRefSetAddress2D.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.POINTER(struct_HIP_ARRAY_DESCRIPTOR), hipDeviceptr_t, size_t]
except AttributeError:
    pass
try:
    hipTexRefSetMaxAnisotropy = _libraries['libamdhip64.so'].hipTexRefSetMaxAnisotropy
    hipTexRefSetMaxAnisotropy.restype = hipError_t
    hipTexRefSetMaxAnisotropy.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipTexRefSetBorderColor = _libraries['libamdhip64.so'].hipTexRefSetBorderColor
    hipTexRefSetBorderColor.restype = hipError_t
    hipTexRefSetBorderColor.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.POINTER(ctypes.c_float)]
except AttributeError:
    pass
try:
    hipTexRefSetMipmapFilterMode = _libraries['libamdhip64.so'].hipTexRefSetMipmapFilterMode
    hipTexRefSetMipmapFilterMode.restype = hipError_t
    hipTexRefSetMipmapFilterMode.argtypes = [ctypes.POINTER(struct_textureReference), hipTextureFilterMode]
except AttributeError:
    pass
try:
    hipTexRefSetMipmapLevelBias = _libraries['libamdhip64.so'].hipTexRefSetMipmapLevelBias
    hipTexRefSetMipmapLevelBias.restype = hipError_t
    hipTexRefSetMipmapLevelBias.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.c_float]
except AttributeError:
    pass
try:
    hipTexRefSetMipmapLevelClamp = _libraries['libamdhip64.so'].hipTexRefSetMipmapLevelClamp
    hipTexRefSetMipmapLevelClamp.restype = hipError_t
    hipTexRefSetMipmapLevelClamp.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    hipTexRefSetMipmappedArray = _libraries['libamdhip64.so'].hipTexRefSetMipmappedArray
    hipTexRefSetMipmappedArray.restype = hipError_t
    hipTexRefSetMipmappedArray.argtypes = [ctypes.POINTER(struct_textureReference), ctypes.POINTER(struct_hipMipmappedArray), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipApiName = _libraries['libamdhip64.so'].hipApiName
    hipApiName.restype = ctypes.POINTER(ctypes.c_char)
    hipApiName.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    hipKernelNameRef = _libraries['libamdhip64.so'].hipKernelNameRef
    hipKernelNameRef.restype = ctypes.POINTER(ctypes.c_char)
    hipKernelNameRef.argtypes = [hipFunction_t]
except AttributeError:
    pass
try:
    hipKernelNameRefByPtr = _libraries['libamdhip64.so'].hipKernelNameRefByPtr
    hipKernelNameRefByPtr.restype = ctypes.POINTER(ctypes.c_char)
    hipKernelNameRefByPtr.argtypes = [ctypes.POINTER(None), hipStream_t]
except AttributeError:
    pass
try:
    hipGetStreamDeviceId = _libraries['libamdhip64.so'].hipGetStreamDeviceId
    hipGetStreamDeviceId.restype = ctypes.c_int32
    hipGetStreamDeviceId.argtypes = [hipStream_t]
except AttributeError:
    pass
try:
    hipStreamBeginCapture = _libraries['libamdhip64.so'].hipStreamBeginCapture
    hipStreamBeginCapture.restype = hipError_t
    hipStreamBeginCapture.argtypes = [hipStream_t, hipStreamCaptureMode]
except AttributeError:
    pass
try:
    hipStreamEndCapture = _libraries['libamdhip64.so'].hipStreamEndCapture
    hipStreamEndCapture.restype = hipError_t
    hipStreamEndCapture.argtypes = [hipStream_t, ctypes.POINTER(ctypes.POINTER(struct_ihipGraph))]
except AttributeError:
    pass
try:
    hipStreamGetCaptureInfo = _libraries['libamdhip64.so'].hipStreamGetCaptureInfo
    hipStreamGetCaptureInfo.restype = hipError_t
    hipStreamGetCaptureInfo.argtypes = [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipStreamGetCaptureInfo_v2 = _libraries['libamdhip64.so'].hipStreamGetCaptureInfo_v2
    hipStreamGetCaptureInfo_v2.restype = hipError_t
    hipStreamGetCaptureInfo_v2.argtypes = [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(struct_ihipGraph)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode))), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipStreamIsCapturing = _libraries['libamdhip64.so'].hipStreamIsCapturing
    hipStreamIsCapturing.restype = hipError_t
    hipStreamIsCapturing.argtypes = [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus)]
except AttributeError:
    pass
try:
    hipStreamUpdateCaptureDependencies = _libraries['libamdhip64.so'].hipStreamUpdateCaptureDependencies
    hipStreamUpdateCaptureDependencies.restype = hipError_t
    hipStreamUpdateCaptureDependencies.argtypes = [hipStream_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipThreadExchangeStreamCaptureMode = _libraries['libamdhip64.so'].hipThreadExchangeStreamCaptureMode
    hipThreadExchangeStreamCaptureMode.restype = hipError_t
    hipThreadExchangeStreamCaptureMode.argtypes = [ctypes.POINTER(hipStreamCaptureMode)]
except AttributeError:
    pass
try:
    hipGraphCreate = _libraries['libamdhip64.so'].hipGraphCreate
    hipGraphCreate.restype = hipError_t
    hipGraphCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipGraph)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphDestroy = _libraries['libamdhip64.so'].hipGraphDestroy
    hipGraphDestroy.restype = hipError_t
    hipGraphDestroy.argtypes = [hipGraph_t]
except AttributeError:
    pass
try:
    hipGraphAddDependencies = _libraries['libamdhip64.so'].hipGraphAddDependencies
    hipGraphAddDependencies.restype = hipError_t
    hipGraphAddDependencies.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t]
except AttributeError:
    pass
try:
    hipGraphRemoveDependencies = _libraries['libamdhip64.so'].hipGraphRemoveDependencies
    hipGraphRemoveDependencies.restype = hipError_t
    hipGraphRemoveDependencies.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t]
except AttributeError:
    pass
try:
    hipGraphGetEdges = _libraries['libamdhip64.so'].hipGraphGetEdges
    hipGraphGetEdges.restype = hipError_t
    hipGraphGetEdges.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipGraphGetNodes = _libraries['libamdhip64.so'].hipGraphGetNodes
    hipGraphGetNodes.restype = hipError_t
    hipGraphGetNodes.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipGraphGetRootNodes = _libraries['libamdhip64.so'].hipGraphGetRootNodes
    hipGraphGetRootNodes.restype = hipError_t
    hipGraphGetRootNodes.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipGraphNodeGetDependencies = _libraries['libamdhip64.so'].hipGraphNodeGetDependencies
    hipGraphNodeGetDependencies.restype = hipError_t
    hipGraphNodeGetDependencies.argtypes = [hipGraphNode_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipGraphNodeGetDependentNodes = _libraries['libamdhip64.so'].hipGraphNodeGetDependentNodes
    hipGraphNodeGetDependentNodes.restype = hipError_t
    hipGraphNodeGetDependentNodes.argtypes = [hipGraphNode_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hipGraphNodeGetType = _libraries['libamdhip64.so'].hipGraphNodeGetType
    hipGraphNodeGetType.restype = hipError_t
    hipGraphNodeGetType.argtypes = [hipGraphNode_t, ctypes.POINTER(hipGraphNodeType)]
except AttributeError:
    pass
try:
    hipGraphDestroyNode = _libraries['libamdhip64.so'].hipGraphDestroyNode
    hipGraphDestroyNode.restype = hipError_t
    hipGraphDestroyNode.argtypes = [hipGraphNode_t]
except AttributeError:
    pass
try:
    hipGraphClone = _libraries['libamdhip64.so'].hipGraphClone
    hipGraphClone.restype = hipError_t
    hipGraphClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipGraph)), hipGraph_t]
except AttributeError:
    pass
try:
    hipGraphNodeFindInClone = _libraries['libamdhip64.so'].hipGraphNodeFindInClone
    hipGraphNodeFindInClone.restype = hipError_t
    hipGraphNodeFindInClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraphNode_t, hipGraph_t]
except AttributeError:
    pass
try:
    hipGraphInstantiate = _libraries['libamdhip64.so'].hipGraphInstantiate
    hipGraphInstantiate.restype = hipError_t
    hipGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphExec)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    hipGraphInstantiateWithFlags = _libraries['libamdhip64.so'].hipGraphInstantiateWithFlags
    hipGraphInstantiateWithFlags.restype = hipError_t
    hipGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphExec)), hipGraph_t, ctypes.c_uint64]
except AttributeError:
    pass
try:
    hipGraphLaunch = _libraries['libamdhip64.so'].hipGraphLaunch
    hipGraphLaunch.restype = hipError_t
    hipGraphLaunch.argtypes = [hipGraphExec_t, hipStream_t]
except AttributeError:
    pass
try:
    hipGraphUpload = _libraries['libamdhip64.so'].hipGraphUpload
    hipGraphUpload.restype = hipError_t
    hipGraphUpload.argtypes = [hipGraphExec_t, hipStream_t]
except AttributeError:
    pass
try:
    hipGraphExecDestroy = _libraries['libamdhip64.so'].hipGraphExecDestroy
    hipGraphExecDestroy.restype = hipError_t
    hipGraphExecDestroy.argtypes = [hipGraphExec_t]
except AttributeError:
    pass
try:
    hipGraphExecUpdate = _libraries['libamdhip64.so'].hipGraphExecUpdate
    hipGraphExecUpdate.restype = hipError_t
    hipGraphExecUpdate.argtypes = [hipGraphExec_t, hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), ctypes.POINTER(hipGraphExecUpdateResult)]
except AttributeError:
    pass
try:
    hipGraphAddKernelNode = _libraries['libamdhip64.so'].hipGraphAddKernelNode
    hipGraphAddKernelNode.restype = hipError_t
    hipGraphAddKernelNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_hipKernelNodeParams)]
except AttributeError:
    pass
try:
    hipGraphKernelNodeGetParams = _libraries['libamdhip64.so'].hipGraphKernelNodeGetParams
    hipGraphKernelNodeGetParams.restype = hipError_t
    hipGraphKernelNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipKernelNodeParams)]
except AttributeError:
    pass
try:
    hipGraphKernelNodeSetParams = _libraries['libamdhip64.so'].hipGraphKernelNodeSetParams
    hipGraphKernelNodeSetParams.restype = hipError_t
    hipGraphKernelNodeSetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipKernelNodeParams)]
except AttributeError:
    pass
try:
    hipGraphExecKernelNodeSetParams = _libraries['libamdhip64.so'].hipGraphExecKernelNodeSetParams
    hipGraphExecKernelNodeSetParams.restype = hipError_t
    hipGraphExecKernelNodeSetParams.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(struct_hipKernelNodeParams)]
except AttributeError:
    pass
try:
    hipDrvGraphAddMemcpyNode = _libraries['FIXME_STUB'].hipDrvGraphAddMemcpyNode
    hipDrvGraphAddMemcpyNode.restype = hipError_t
    hipDrvGraphAddMemcpyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_HIP_MEMCPY3D), hipCtx_t]
except AttributeError:
    pass
try:
    hipGraphAddMemcpyNode = _libraries['libamdhip64.so'].hipGraphAddMemcpyNode
    hipGraphAddMemcpyNode.restype = hipError_t
    hipGraphAddMemcpyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_hipMemcpy3DParms)]
except AttributeError:
    pass
try:
    hipGraphMemcpyNodeGetParams = _libraries['libamdhip64.so'].hipGraphMemcpyNodeGetParams
    hipGraphMemcpyNodeGetParams.restype = hipError_t
    hipGraphMemcpyNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipMemcpy3DParms)]
except AttributeError:
    pass
try:
    hipGraphMemcpyNodeSetParams = _libraries['libamdhip64.so'].hipGraphMemcpyNodeSetParams
    hipGraphMemcpyNodeSetParams.restype = hipError_t
    hipGraphMemcpyNodeSetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipMemcpy3DParms)]
except AttributeError:
    pass
try:
    hipGraphKernelNodeSetAttribute = _libraries['libamdhip64.so'].hipGraphKernelNodeSetAttribute
    hipGraphKernelNodeSetAttribute.restype = hipError_t
    hipGraphKernelNodeSetAttribute.argtypes = [hipGraphNode_t, hipKernelNodeAttrID, ctypes.POINTER(union_hipKernelNodeAttrValue)]
except AttributeError:
    pass
try:
    hipGraphKernelNodeGetAttribute = _libraries['libamdhip64.so'].hipGraphKernelNodeGetAttribute
    hipGraphKernelNodeGetAttribute.restype = hipError_t
    hipGraphKernelNodeGetAttribute.argtypes = [hipGraphNode_t, hipKernelNodeAttrID, ctypes.POINTER(union_hipKernelNodeAttrValue)]
except AttributeError:
    pass
try:
    hipGraphExecMemcpyNodeSetParams = _libraries['libamdhip64.so'].hipGraphExecMemcpyNodeSetParams
    hipGraphExecMemcpyNodeSetParams.restype = hipError_t
    hipGraphExecMemcpyNodeSetParams.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(struct_hipMemcpy3DParms)]
except AttributeError:
    pass
try:
    hipGraphAddMemcpyNode1D = _libraries['libamdhip64.so'].hipGraphAddMemcpyNode1D
    hipGraphAddMemcpyNode1D.restype = hipError_t
    hipGraphAddMemcpyNode1D.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphMemcpyNodeSetParams1D = _libraries['libamdhip64.so'].hipGraphMemcpyNodeSetParams1D
    hipGraphMemcpyNodeSetParams1D.restype = hipError_t
    hipGraphMemcpyNodeSetParams1D.argtypes = [hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphExecMemcpyNodeSetParams1D = _libraries['libamdhip64.so'].hipGraphExecMemcpyNodeSetParams1D
    hipGraphExecMemcpyNodeSetParams1D.restype = hipError_t
    hipGraphExecMemcpyNodeSetParams1D.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphAddMemcpyNodeFromSymbol = _libraries['libamdhip64.so'].hipGraphAddMemcpyNodeFromSymbol
    hipGraphAddMemcpyNodeFromSymbol.restype = hipError_t
    hipGraphAddMemcpyNodeFromSymbol.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphMemcpyNodeSetParamsFromSymbol = _libraries['libamdhip64.so'].hipGraphMemcpyNodeSetParamsFromSymbol
    hipGraphMemcpyNodeSetParamsFromSymbol.restype = hipError_t
    hipGraphMemcpyNodeSetParamsFromSymbol.argtypes = [hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphExecMemcpyNodeSetParamsFromSymbol = _libraries['libamdhip64.so'].hipGraphExecMemcpyNodeSetParamsFromSymbol
    hipGraphExecMemcpyNodeSetParamsFromSymbol.restype = hipError_t
    hipGraphExecMemcpyNodeSetParamsFromSymbol.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphAddMemcpyNodeToSymbol = _libraries['libamdhip64.so'].hipGraphAddMemcpyNodeToSymbol
    hipGraphAddMemcpyNodeToSymbol.restype = hipError_t
    hipGraphAddMemcpyNodeToSymbol.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphMemcpyNodeSetParamsToSymbol = _libraries['libamdhip64.so'].hipGraphMemcpyNodeSetParamsToSymbol
    hipGraphMemcpyNodeSetParamsToSymbol.restype = hipError_t
    hipGraphMemcpyNodeSetParamsToSymbol.argtypes = [hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphExecMemcpyNodeSetParamsToSymbol = _libraries['libamdhip64.so'].hipGraphExecMemcpyNodeSetParamsToSymbol
    hipGraphExecMemcpyNodeSetParamsToSymbol.restype = hipError_t
    hipGraphExecMemcpyNodeSetParamsToSymbol.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, hipMemcpyKind]
except AttributeError:
    pass
try:
    hipGraphAddMemsetNode = _libraries['libamdhip64.so'].hipGraphAddMemsetNode
    hipGraphAddMemsetNode.restype = hipError_t
    hipGraphAddMemsetNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_hipMemsetParams)]
except AttributeError:
    pass
try:
    hipGraphMemsetNodeGetParams = _libraries['libamdhip64.so'].hipGraphMemsetNodeGetParams
    hipGraphMemsetNodeGetParams.restype = hipError_t
    hipGraphMemsetNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipMemsetParams)]
except AttributeError:
    pass
try:
    hipGraphMemsetNodeSetParams = _libraries['libamdhip64.so'].hipGraphMemsetNodeSetParams
    hipGraphMemsetNodeSetParams.restype = hipError_t
    hipGraphMemsetNodeSetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipMemsetParams)]
except AttributeError:
    pass
try:
    hipGraphExecMemsetNodeSetParams = _libraries['libamdhip64.so'].hipGraphExecMemsetNodeSetParams
    hipGraphExecMemsetNodeSetParams.restype = hipError_t
    hipGraphExecMemsetNodeSetParams.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(struct_hipMemsetParams)]
except AttributeError:
    pass
try:
    hipGraphAddHostNode = _libraries['libamdhip64.so'].hipGraphAddHostNode
    hipGraphAddHostNode.restype = hipError_t
    hipGraphAddHostNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_hipHostNodeParams)]
except AttributeError:
    pass
try:
    hipGraphHostNodeGetParams = _libraries['libamdhip64.so'].hipGraphHostNodeGetParams
    hipGraphHostNodeGetParams.restype = hipError_t
    hipGraphHostNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipHostNodeParams)]
except AttributeError:
    pass
try:
    hipGraphHostNodeSetParams = _libraries['libamdhip64.so'].hipGraphHostNodeSetParams
    hipGraphHostNodeSetParams.restype = hipError_t
    hipGraphHostNodeSetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipHostNodeParams)]
except AttributeError:
    pass
try:
    hipGraphExecHostNodeSetParams = _libraries['libamdhip64.so'].hipGraphExecHostNodeSetParams
    hipGraphExecHostNodeSetParams.restype = hipError_t
    hipGraphExecHostNodeSetParams.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(struct_hipHostNodeParams)]
except AttributeError:
    pass
try:
    hipGraphAddChildGraphNode = _libraries['libamdhip64.so'].hipGraphAddChildGraphNode
    hipGraphAddChildGraphNode.restype = hipError_t
    hipGraphAddChildGraphNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, hipGraph_t]
except AttributeError:
    pass
try:
    hipGraphChildGraphNodeGetGraph = _libraries['libamdhip64.so'].hipGraphChildGraphNodeGetGraph
    hipGraphChildGraphNodeGetGraph.restype = hipError_t
    hipGraphChildGraphNodeGetGraph.argtypes = [hipGraphNode_t, ctypes.POINTER(ctypes.POINTER(struct_ihipGraph))]
except AttributeError:
    pass
try:
    hipGraphExecChildGraphNodeSetParams = _libraries['libamdhip64.so'].hipGraphExecChildGraphNodeSetParams
    hipGraphExecChildGraphNodeSetParams.restype = hipError_t
    hipGraphExecChildGraphNodeSetParams.argtypes = [hipGraphExec_t, hipGraphNode_t, hipGraph_t]
except AttributeError:
    pass
try:
    hipGraphAddEmptyNode = _libraries['libamdhip64.so'].hipGraphAddEmptyNode
    hipGraphAddEmptyNode.restype = hipError_t
    hipGraphAddEmptyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t]
except AttributeError:
    pass
try:
    hipGraphAddEventRecordNode = _libraries['libamdhip64.so'].hipGraphAddEventRecordNode
    hipGraphAddEventRecordNode.restype = hipError_t
    hipGraphAddEventRecordNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphEventRecordNodeGetEvent = _libraries['libamdhip64.so'].hipGraphEventRecordNodeGetEvent
    hipGraphEventRecordNodeGetEvent.restype = hipError_t
    hipGraphEventRecordNodeGetEvent.argtypes = [hipGraphNode_t, ctypes.POINTER(ctypes.POINTER(struct_ihipEvent_t))]
except AttributeError:
    pass
try:
    hipGraphEventRecordNodeSetEvent = _libraries['libamdhip64.so'].hipGraphEventRecordNodeSetEvent
    hipGraphEventRecordNodeSetEvent.restype = hipError_t
    hipGraphEventRecordNodeSetEvent.argtypes = [hipGraphNode_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphExecEventRecordNodeSetEvent = _libraries['libamdhip64.so'].hipGraphExecEventRecordNodeSetEvent
    hipGraphExecEventRecordNodeSetEvent.restype = hipError_t
    hipGraphExecEventRecordNodeSetEvent.argtypes = [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphAddEventWaitNode = _libraries['libamdhip64.so'].hipGraphAddEventWaitNode
    hipGraphAddEventWaitNode.restype = hipError_t
    hipGraphAddEventWaitNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphEventWaitNodeGetEvent = _libraries['libamdhip64.so'].hipGraphEventWaitNodeGetEvent
    hipGraphEventWaitNodeGetEvent.restype = hipError_t
    hipGraphEventWaitNodeGetEvent.argtypes = [hipGraphNode_t, ctypes.POINTER(ctypes.POINTER(struct_ihipEvent_t))]
except AttributeError:
    pass
try:
    hipGraphEventWaitNodeSetEvent = _libraries['libamdhip64.so'].hipGraphEventWaitNodeSetEvent
    hipGraphEventWaitNodeSetEvent.restype = hipError_t
    hipGraphEventWaitNodeSetEvent.argtypes = [hipGraphNode_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphExecEventWaitNodeSetEvent = _libraries['libamdhip64.so'].hipGraphExecEventWaitNodeSetEvent
    hipGraphExecEventWaitNodeSetEvent.restype = hipError_t
    hipGraphExecEventWaitNodeSetEvent.argtypes = [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError:
    pass
try:
    hipGraphAddMemAllocNode = _libraries['libamdhip64.so'].hipGraphAddMemAllocNode
    hipGraphAddMemAllocNode.restype = hipError_t
    hipGraphAddMemAllocNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(struct_hipMemAllocNodeParams)]
except AttributeError:
    pass
try:
    hipGraphMemAllocNodeGetParams = _libraries['libamdhip64.so'].hipGraphMemAllocNodeGetParams
    hipGraphMemAllocNodeGetParams.restype = hipError_t
    hipGraphMemAllocNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(struct_hipMemAllocNodeParams)]
except AttributeError:
    pass
try:
    hipGraphAddMemFreeNode = _libraries['libamdhip64.so'].hipGraphAddMemFreeNode
    hipGraphAddMemFreeNode.restype = hipError_t
    hipGraphAddMemFreeNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), hipGraph_t, ctypes.POINTER(ctypes.POINTER(struct_hipGraphNode)), size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipGraphMemFreeNodeGetParams = _libraries['libamdhip64.so'].hipGraphMemFreeNodeGetParams
    hipGraphMemFreeNodeGetParams.restype = hipError_t
    hipGraphMemFreeNodeGetParams.argtypes = [hipGraphNode_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipDeviceGetGraphMemAttribute = _libraries['libamdhip64.so'].hipDeviceGetGraphMemAttribute
    hipDeviceGetGraphMemAttribute.restype = hipError_t
    hipDeviceGetGraphMemAttribute.argtypes = [ctypes.c_int32, hipGraphMemAttributeType, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipDeviceSetGraphMemAttribute = _libraries['libamdhip64.so'].hipDeviceSetGraphMemAttribute
    hipDeviceSetGraphMemAttribute.restype = hipError_t
    hipDeviceSetGraphMemAttribute.argtypes = [ctypes.c_int32, hipGraphMemAttributeType, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipDeviceGraphMemTrim = _libraries['libamdhip64.so'].hipDeviceGraphMemTrim
    hipDeviceGraphMemTrim.restype = hipError_t
    hipDeviceGraphMemTrim.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    hipUserObjectCreate = _libraries['libamdhip64.so'].hipUserObjectCreate
    hipUserObjectCreate.restype = hipError_t
    hipUserObjectCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipUserObject)), ctypes.POINTER(None), hipHostFn_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipUserObjectRelease = _libraries['libamdhip64.so'].hipUserObjectRelease
    hipUserObjectRelease.restype = hipError_t
    hipUserObjectRelease.argtypes = [hipUserObject_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipUserObjectRetain = _libraries['libamdhip64.so'].hipUserObjectRetain
    hipUserObjectRetain.restype = hipError_t
    hipUserObjectRetain.argtypes = [hipUserObject_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphRetainUserObject = _libraries['libamdhip64.so'].hipGraphRetainUserObject
    hipGraphRetainUserObject.restype = hipError_t
    hipGraphRetainUserObject.argtypes = [hipGraph_t, hipUserObject_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphReleaseUserObject = _libraries['libamdhip64.so'].hipGraphReleaseUserObject
    hipGraphReleaseUserObject.restype = hipError_t
    hipGraphReleaseUserObject.argtypes = [hipGraph_t, hipUserObject_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphDebugDotPrint = _libraries['libamdhip64.so'].hipGraphDebugDotPrint
    hipGraphDebugDotPrint.restype = hipError_t
    hipGraphDebugDotPrint.argtypes = [hipGraph_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphKernelNodeCopyAttributes = _libraries['libamdhip64.so'].hipGraphKernelNodeCopyAttributes
    hipGraphKernelNodeCopyAttributes.restype = hipError_t
    hipGraphKernelNodeCopyAttributes.argtypes = [hipGraphNode_t, hipGraphNode_t]
except AttributeError:
    pass
try:
    hipGraphNodeSetEnabled = _libraries['libamdhip64.so'].hipGraphNodeSetEnabled
    hipGraphNodeSetEnabled.restype = hipError_t
    hipGraphNodeSetEnabled.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphNodeGetEnabled = _libraries['libamdhip64.so'].hipGraphNodeGetEnabled
    hipGraphNodeGetEnabled.restype = hipError_t
    hipGraphNodeGetEnabled.argtypes = [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hipMemAddressFree = _libraries['libamdhip64.so'].hipMemAddressFree
    hipMemAddressFree.restype = hipError_t
    hipMemAddressFree.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipMemAddressReserve = _libraries['libamdhip64.so'].hipMemAddressReserve
    hipMemAddressReserve.restype = hipError_t
    hipMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, size_t, ctypes.POINTER(None), ctypes.c_uint64]
except AttributeError:
    pass
try:
    hipMemCreate = _libraries['libamdhip64.so'].hipMemCreate
    hipMemCreate.restype = hipError_t
    hipMemCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemGenericAllocationHandle)), size_t, ctypes.POINTER(struct_hipMemAllocationProp), ctypes.c_uint64]
except AttributeError:
    pass
try:
    hipMemExportToShareableHandle = _libraries['libamdhip64.so'].hipMemExportToShareableHandle
    hipMemExportToShareableHandle.restype = hipError_t
    hipMemExportToShareableHandle.argtypes = [ctypes.POINTER(None), hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    hipMemGetAccess = _libraries['libamdhip64.so'].hipMemGetAccess
    hipMemGetAccess.restype = hipError_t
    hipMemGetAccess.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_hipMemLocation), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemGetAllocationGranularity = _libraries['libamdhip64.so'].hipMemGetAllocationGranularity
    hipMemGetAllocationGranularity.restype = hipError_t
    hipMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_hipMemAllocationProp), hipMemAllocationGranularity_flags]
except AttributeError:
    pass
try:
    hipMemGetAllocationPropertiesFromHandle = _libraries['libamdhip64.so'].hipMemGetAllocationPropertiesFromHandle
    hipMemGetAllocationPropertiesFromHandle.restype = hipError_t
    hipMemGetAllocationPropertiesFromHandle.argtypes = [ctypes.POINTER(struct_hipMemAllocationProp), hipMemGenericAllocationHandle_t]
except AttributeError:
    pass
try:
    hipMemImportFromShareableHandle = _libraries['libamdhip64.so'].hipMemImportFromShareableHandle
    hipMemImportFromShareableHandle.restype = hipError_t
    hipMemImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemGenericAllocationHandle)), ctypes.POINTER(None), hipMemAllocationHandleType]
except AttributeError:
    pass
try:
    hipMemMap = _libraries['libamdhip64.so'].hipMemMap
    hipMemMap.restype = hipError_t
    hipMemMap.argtypes = [ctypes.POINTER(None), size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_uint64]
except AttributeError:
    pass
try:
    hipMemMapArrayAsync = _libraries['libamdhip64.so'].hipMemMapArrayAsync
    hipMemMapArrayAsync.restype = hipError_t
    hipMemMapArrayAsync.argtypes = [ctypes.POINTER(struct_hipArrayMapInfo), ctypes.c_uint32, hipStream_t]
except AttributeError:
    pass
try:
    hipMemRelease = _libraries['libamdhip64.so'].hipMemRelease
    hipMemRelease.restype = hipError_t
    hipMemRelease.argtypes = [hipMemGenericAllocationHandle_t]
except AttributeError:
    pass
try:
    hipMemRetainAllocationHandle = _libraries['libamdhip64.so'].hipMemRetainAllocationHandle
    hipMemRetainAllocationHandle.restype = hipError_t
    hipMemRetainAllocationHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ihipMemGenericAllocationHandle)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hipMemSetAccess = _libraries['libamdhip64.so'].hipMemSetAccess
    hipMemSetAccess.restype = hipError_t
    hipMemSetAccess.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hipMemAccessDesc), size_t]
except AttributeError:
    pass
try:
    hipMemUnmap = _libraries['libamdhip64.so'].hipMemUnmap
    hipMemUnmap.restype = hipError_t
    hipMemUnmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hipGraphicsMapResources = _libraries['libamdhip64.so'].hipGraphicsMapResources
    hipGraphicsMapResources.restype = hipError_t
    hipGraphicsMapResources.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(struct__hipGraphicsResource)), hipStream_t]
except AttributeError:
    pass
try:
    hipGraphicsSubResourceGetMappedArray = _libraries['libamdhip64.so'].hipGraphicsSubResourceGetMappedArray
    hipGraphicsSubResourceGetMappedArray.restype = hipError_t
    hipGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_hipArray)), hipGraphicsResource_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    hipGraphicsResourceGetMappedPointer = _libraries['libamdhip64.so'].hipGraphicsResourceGetMappedPointer
    hipGraphicsResourceGetMappedPointer.restype = hipError_t
    hipGraphicsResourceGetMappedPointer.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), hipGraphicsResource_t]
except AttributeError:
    pass
try:
    hipGraphicsUnmapResources = _libraries['libamdhip64.so'].hipGraphicsUnmapResources
    hipGraphicsUnmapResources.restype = hipError_t
    hipGraphicsUnmapResources.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(struct__hipGraphicsResource)), hipStream_t]
except AttributeError:
    pass
try:
    hipGraphicsUnregisterResource = _libraries['libamdhip64.so'].hipGraphicsUnregisterResource
    hipGraphicsUnregisterResource.restype = hipError_t
    hipGraphicsUnregisterResource.argtypes = [hipGraphicsResource_t]
except AttributeError:
    pass
class struct___hip_surface(Structure):
    pass

try:
    hipCreateSurfaceObject = _libraries['libamdhip64.so'].hipCreateSurfaceObject
    hipCreateSurfaceObject.restype = hipError_t
    hipCreateSurfaceObject.argtypes = [ctypes.POINTER(ctypes.POINTER(struct___hip_surface)), ctypes.POINTER(struct_hipResourceDesc)]
except AttributeError:
    pass
hipSurfaceObject_t = ctypes.POINTER(struct___hip_surface)
try:
    hipDestroySurfaceObject = _libraries['libamdhip64.so'].hipDestroySurfaceObject
    hipDestroySurfaceObject.restype = hipError_t
    hipDestroySurfaceObject.argtypes = [hipSurfaceObject_t]
except AttributeError:
    pass
try:
    hipExtModuleLaunchKernel = _libraries['FIXME_STUB'].hipExtModuleLaunchKernel
    hipExtModuleLaunchKernel.restype = hipError_t
    hipExtModuleLaunchKernel.argtypes = [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None)), hipEvent_t, hipEvent_t, uint32_t]
except AttributeError:
    pass
try:
    hipHccModuleLaunchKernel = _libraries['FIXME_STUB'].hipHccModuleLaunchKernel
    hipHccModuleLaunchKernel.restype = hipError_t
    hipHccModuleLaunchKernel.argtypes = [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None)), hipEvent_t, hipEvent_t]
except AttributeError:
    pass

# values for enumeration 'hiprtcResult'
hiprtcResult__enumvalues = {
    0: 'HIPRTC_SUCCESS',
    1: 'HIPRTC_ERROR_OUT_OF_MEMORY',
    2: 'HIPRTC_ERROR_PROGRAM_CREATION_FAILURE',
    3: 'HIPRTC_ERROR_INVALID_INPUT',
    4: 'HIPRTC_ERROR_INVALID_PROGRAM',
    5: 'HIPRTC_ERROR_INVALID_OPTION',
    6: 'HIPRTC_ERROR_COMPILATION',
    7: 'HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    8: 'HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    9: 'HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    10: 'HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    11: 'HIPRTC_ERROR_INTERNAL_ERROR',
    100: 'HIPRTC_ERROR_LINKING',
}
HIPRTC_SUCCESS = 0
HIPRTC_ERROR_OUT_OF_MEMORY = 1
HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
HIPRTC_ERROR_INVALID_INPUT = 3
HIPRTC_ERROR_INVALID_PROGRAM = 4
HIPRTC_ERROR_INVALID_OPTION = 5
HIPRTC_ERROR_COMPILATION = 6
HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
HIPRTC_ERROR_INTERNAL_ERROR = 11
HIPRTC_ERROR_LINKING = 100
hiprtcResult = ctypes.c_uint32 # enum

# values for enumeration 'hiprtcJIT_option'
hiprtcJIT_option__enumvalues = {
    0: 'HIPRTC_JIT_MAX_REGISTERS',
    1: 'HIPRTC_JIT_THREADS_PER_BLOCK',
    2: 'HIPRTC_JIT_WALL_TIME',
    3: 'HIPRTC_JIT_INFO_LOG_BUFFER',
    4: 'HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES',
    5: 'HIPRTC_JIT_ERROR_LOG_BUFFER',
    6: 'HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES',
    7: 'HIPRTC_JIT_OPTIMIZATION_LEVEL',
    8: 'HIPRTC_JIT_TARGET_FROM_HIPCONTEXT',
    9: 'HIPRTC_JIT_TARGET',
    10: 'HIPRTC_JIT_FALLBACK_STRATEGY',
    11: 'HIPRTC_JIT_GENERATE_DEBUG_INFO',
    12: 'HIPRTC_JIT_LOG_VERBOSE',
    13: 'HIPRTC_JIT_GENERATE_LINE_INFO',
    14: 'HIPRTC_JIT_CACHE_MODE',
    15: 'HIPRTC_JIT_NEW_SM3X_OPT',
    16: 'HIPRTC_JIT_FAST_COMPILE',
    17: 'HIPRTC_JIT_GLOBAL_SYMBOL_NAMES',
    18: 'HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS',
    19: 'HIPRTC_JIT_GLOBAL_SYMBOL_COUNT',
    20: 'HIPRTC_JIT_LTO',
    21: 'HIPRTC_JIT_FTZ',
    22: 'HIPRTC_JIT_PREC_DIV',
    23: 'HIPRTC_JIT_PREC_SQRT',
    24: 'HIPRTC_JIT_FMA',
    25: 'HIPRTC_JIT_NUM_OPTIONS',
    10000: 'HIPRTC_JIT_IR_TO_ISA_OPT_EXT',
    10001: 'HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT',
}
HIPRTC_JIT_MAX_REGISTERS = 0
HIPRTC_JIT_THREADS_PER_BLOCK = 1
HIPRTC_JIT_WALL_TIME = 2
HIPRTC_JIT_INFO_LOG_BUFFER = 3
HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
HIPRTC_JIT_ERROR_LOG_BUFFER = 5
HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
HIPRTC_JIT_OPTIMIZATION_LEVEL = 7
HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = 8
HIPRTC_JIT_TARGET = 9
HIPRTC_JIT_FALLBACK_STRATEGY = 10
HIPRTC_JIT_GENERATE_DEBUG_INFO = 11
HIPRTC_JIT_LOG_VERBOSE = 12
HIPRTC_JIT_GENERATE_LINE_INFO = 13
HIPRTC_JIT_CACHE_MODE = 14
HIPRTC_JIT_NEW_SM3X_OPT = 15
HIPRTC_JIT_FAST_COMPILE = 16
HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = 17
HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = 18
HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = 19
HIPRTC_JIT_LTO = 20
HIPRTC_JIT_FTZ = 21
HIPRTC_JIT_PREC_DIV = 22
HIPRTC_JIT_PREC_SQRT = 23
HIPRTC_JIT_FMA = 24
HIPRTC_JIT_NUM_OPTIONS = 25
HIPRTC_JIT_IR_TO_ISA_OPT_EXT = 10000
HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT = 10001
hiprtcJIT_option = ctypes.c_uint32 # enum

# values for enumeration 'hiprtcJITInputType'
hiprtcJITInputType__enumvalues = {
    0: 'HIPRTC_JIT_INPUT_CUBIN',
    1: 'HIPRTC_JIT_INPUT_PTX',
    2: 'HIPRTC_JIT_INPUT_FATBINARY',
    3: 'HIPRTC_JIT_INPUT_OBJECT',
    4: 'HIPRTC_JIT_INPUT_LIBRARY',
    5: 'HIPRTC_JIT_INPUT_NVVM',
    6: 'HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES',
    100: 'HIPRTC_JIT_INPUT_LLVM_BITCODE',
    101: 'HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE',
    102: 'HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE',
    9: 'HIPRTC_JIT_NUM_INPUT_TYPES',
}
HIPRTC_JIT_INPUT_CUBIN = 0
HIPRTC_JIT_INPUT_PTX = 1
HIPRTC_JIT_INPUT_FATBINARY = 2
HIPRTC_JIT_INPUT_OBJECT = 3
HIPRTC_JIT_INPUT_LIBRARY = 4
HIPRTC_JIT_INPUT_NVVM = 5
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = 6
HIPRTC_JIT_INPUT_LLVM_BITCODE = 100
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102
HIPRTC_JIT_NUM_INPUT_TYPES = 9
hiprtcJITInputType = ctypes.c_uint32 # enum
class struct_ihiprtcLinkState(Structure):
    pass

hiprtcLinkState = ctypes.POINTER(struct_ihiprtcLinkState)
try:
    hiprtcGetErrorString = _libraries['libamdhip64.so'].hiprtcGetErrorString
    hiprtcGetErrorString.restype = ctypes.POINTER(ctypes.c_char)
    hiprtcGetErrorString.argtypes = [hiprtcResult]
except AttributeError:
    pass
try:
    hiprtcVersion = _libraries['libamdhip64.so'].hiprtcVersion
    hiprtcVersion.restype = hiprtcResult
    hiprtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
class struct__hiprtcProgram(Structure):
    pass

hiprtcProgram = ctypes.POINTER(struct__hiprtcProgram)
try:
    hiprtcAddNameExpression = _libraries['libamdhip64.so'].hiprtcAddNameExpression
    hiprtcAddNameExpression.restype = hiprtcResult
    hiprtcAddNameExpression.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hiprtcCompileProgram = _libraries['libamdhip64.so'].hiprtcCompileProgram
    hiprtcCompileProgram.restype = hiprtcResult
    hiprtcCompileProgram.argtypes = [hiprtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hiprtcCreateProgram = _libraries['libamdhip64.so'].hiprtcCreateProgram
    hiprtcCreateProgram.restype = hiprtcResult
    hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__hiprtcProgram)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hiprtcDestroyProgram = _libraries['libamdhip64.so'].hiprtcDestroyProgram
    hiprtcDestroyProgram.restype = hiprtcResult
    hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__hiprtcProgram))]
except AttributeError:
    pass
try:
    hiprtcGetLoweredName = _libraries['libamdhip64.so'].hiprtcGetLoweredName
    hiprtcGetLoweredName.restype = hiprtcResult
    hiprtcGetLoweredName.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hiprtcGetProgramLog = _libraries['libamdhip64.so'].hiprtcGetProgramLog
    hiprtcGetProgramLog.restype = hiprtcResult
    hiprtcGetProgramLog.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hiprtcGetProgramLogSize = _libraries['libamdhip64.so'].hiprtcGetProgramLogSize
    hiprtcGetProgramLogSize.restype = hiprtcResult
    hiprtcGetProgramLogSize.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hiprtcGetCode = _libraries['libamdhip64.so'].hiprtcGetCode
    hiprtcGetCode.restype = hiprtcResult
    hiprtcGetCode.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hiprtcGetCodeSize = _libraries['libamdhip64.so'].hiprtcGetCodeSize
    hiprtcGetCodeSize.restype = hiprtcResult
    hiprtcGetCodeSize.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hiprtcGetBitcode = _libraries['libamdhip64.so'].hiprtcGetBitcode
    hiprtcGetBitcode.restype = hiprtcResult
    hiprtcGetBitcode.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    hiprtcGetBitcodeSize = _libraries['libamdhip64.so'].hiprtcGetBitcodeSize
    hiprtcGetBitcodeSize.restype = hiprtcResult
    hiprtcGetBitcodeSize.argtypes = [hiprtcProgram, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hiprtcLinkCreate = _libraries['libamdhip64.so'].hiprtcLinkCreate
    hiprtcLinkCreate.restype = hiprtcResult
    hiprtcLinkCreate.argtypes = [ctypes.c_uint32, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(struct_ihiprtcLinkState))]
except AttributeError:
    pass
try:
    hiprtcLinkAddFile = _libraries['libamdhip64.so'].hiprtcLinkAddFile
    hiprtcLinkAddFile.restype = hiprtcResult
    hiprtcLinkAddFile.argtypes = [hiprtcLinkState, hiprtcJITInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hiprtcLinkAddData = _libraries['libamdhip64.so'].hiprtcLinkAddData
    hiprtcLinkAddData.restype = hiprtcResult
    hiprtcLinkAddData.argtypes = [hiprtcLinkState, hiprtcJITInputType, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hiprtcLinkComplete = _libraries['libamdhip64.so'].hiprtcLinkComplete
    hiprtcLinkComplete.restype = hiprtcResult
    hiprtcLinkComplete.argtypes = [hiprtcLinkState, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hiprtcLinkDestroy = _libraries['libamdhip64.so'].hiprtcLinkDestroy
    hiprtcLinkDestroy.restype = hiprtcResult
    hiprtcLinkDestroy.argtypes = [hiprtcLinkState]
except AttributeError:
    pass
__all__ = \
    ['HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE',
    'HIPRTC_ERROR_COMPILATION', 'HIPRTC_ERROR_INTERNAL_ERROR',
    'HIPRTC_ERROR_INVALID_INPUT', 'HIPRTC_ERROR_INVALID_OPTION',
    'HIPRTC_ERROR_INVALID_PROGRAM', 'HIPRTC_ERROR_LINKING',
    'HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID',
    'HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION',
    'HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION',
    'HIPRTC_ERROR_OUT_OF_MEMORY',
    'HIPRTC_ERROR_PROGRAM_CREATION_FAILURE', 'HIPRTC_JIT_CACHE_MODE',
    'HIPRTC_JIT_ERROR_LOG_BUFFER',
    'HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES',
    'HIPRTC_JIT_FALLBACK_STRATEGY', 'HIPRTC_JIT_FAST_COMPILE',
    'HIPRTC_JIT_FMA', 'HIPRTC_JIT_FTZ',
    'HIPRTC_JIT_GENERATE_DEBUG_INFO', 'HIPRTC_JIT_GENERATE_LINE_INFO',
    'HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS',
    'HIPRTC_JIT_GLOBAL_SYMBOL_COUNT',
    'HIPRTC_JIT_GLOBAL_SYMBOL_NAMES', 'HIPRTC_JIT_INFO_LOG_BUFFER',
    'HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 'HIPRTC_JIT_INPUT_CUBIN',
    'HIPRTC_JIT_INPUT_FATBINARY', 'HIPRTC_JIT_INPUT_LIBRARY',
    'HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE',
    'HIPRTC_JIT_INPUT_LLVM_BITCODE',
    'HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE', 'HIPRTC_JIT_INPUT_NVVM',
    'HIPRTC_JIT_INPUT_OBJECT', 'HIPRTC_JIT_INPUT_PTX',
    'HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT',
    'HIPRTC_JIT_IR_TO_ISA_OPT_EXT', 'HIPRTC_JIT_LOG_VERBOSE',
    'HIPRTC_JIT_LTO', 'HIPRTC_JIT_MAX_REGISTERS',
    'HIPRTC_JIT_NEW_SM3X_OPT', 'HIPRTC_JIT_NUM_INPUT_TYPES',
    'HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES', 'HIPRTC_JIT_NUM_OPTIONS',
    'HIPRTC_JIT_OPTIMIZATION_LEVEL', 'HIPRTC_JIT_PREC_DIV',
    'HIPRTC_JIT_PREC_SQRT', 'HIPRTC_JIT_TARGET',
    'HIPRTC_JIT_TARGET_FROM_HIPCONTEXT',
    'HIPRTC_JIT_THREADS_PER_BLOCK', 'HIPRTC_JIT_WALL_TIME',
    'HIPRTC_SUCCESS', 'HIP_AD_FORMAT_FLOAT', 'HIP_AD_FORMAT_HALF',
    'HIP_AD_FORMAT_SIGNED_INT16', 'HIP_AD_FORMAT_SIGNED_INT32',
    'HIP_AD_FORMAT_SIGNED_INT8', 'HIP_AD_FORMAT_UNSIGNED_INT16',
    'HIP_AD_FORMAT_UNSIGNED_INT32', 'HIP_AD_FORMAT_UNSIGNED_INT8',
    'HIP_ARRAY3D_DESCRIPTOR', 'HIP_ARRAY_DESCRIPTOR',
    'HIP_ERROR_INVALID_VALUE', 'HIP_ERROR_LAUNCH_OUT_OF_RESOURCES',
    'HIP_ERROR_NOT_INITIALIZED', 'HIP_FUNC_ATTRIBUTE_BINARY_VERSION',
    'HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    'HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    'HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 'HIP_FUNC_ATTRIBUTE_MAX',
    'HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    'HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'HIP_FUNC_ATTRIBUTE_NUM_REGS',
    'HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    'HIP_FUNC_ATTRIBUTE_PTX_VERSION',
    'HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', 'HIP_MEMCPY3D',
    'HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    'HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    'HIP_POINTER_ATTRIBUTE_BUFFER_ID',
    'HIP_POINTER_ATTRIBUTE_CONTEXT',
    'HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    'HIP_POINTER_ATTRIBUTE_DEVICE_POINTER',
    'HIP_POINTER_ATTRIBUTE_HOST_POINTER',
    'HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    'HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE',
    'HIP_POINTER_ATTRIBUTE_IS_MANAGED',
    'HIP_POINTER_ATTRIBUTE_MAPPED',
    'HIP_POINTER_ATTRIBUTE_MEMORY_TYPE',
    'HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
    'HIP_POINTER_ATTRIBUTE_P2P_TOKENS',
    'HIP_POINTER_ATTRIBUTE_RANGE_SIZE',
    'HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    'HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS', 'HIP_RESOURCE_DESC',
    'HIP_RESOURCE_TYPE_ARRAY', 'HIP_RESOURCE_TYPE_LINEAR',
    'HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 'HIP_RESOURCE_TYPE_PITCH2D',
    'HIP_RESOURCE_VIEW_DESC', 'HIP_RES_VIEW_FORMAT_FLOAT_1X16',
    'HIP_RES_VIEW_FORMAT_FLOAT_1X32',
    'HIP_RES_VIEW_FORMAT_FLOAT_2X16',
    'HIP_RES_VIEW_FORMAT_FLOAT_2X32',
    'HIP_RES_VIEW_FORMAT_FLOAT_4X16',
    'HIP_RES_VIEW_FORMAT_FLOAT_4X32', 'HIP_RES_VIEW_FORMAT_NONE',
    'HIP_RES_VIEW_FORMAT_SIGNED_BC4',
    'HIP_RES_VIEW_FORMAT_SIGNED_BC5',
    'HIP_RES_VIEW_FORMAT_SIGNED_BC6H',
    'HIP_RES_VIEW_FORMAT_SINT_1X16', 'HIP_RES_VIEW_FORMAT_SINT_1X32',
    'HIP_RES_VIEW_FORMAT_SINT_1X8', 'HIP_RES_VIEW_FORMAT_SINT_2X16',
    'HIP_RES_VIEW_FORMAT_SINT_2X32', 'HIP_RES_VIEW_FORMAT_SINT_2X8',
    'HIP_RES_VIEW_FORMAT_SINT_4X16', 'HIP_RES_VIEW_FORMAT_SINT_4X32',
    'HIP_RES_VIEW_FORMAT_SINT_4X8', 'HIP_RES_VIEW_FORMAT_UINT_1X16',
    'HIP_RES_VIEW_FORMAT_UINT_1X32', 'HIP_RES_VIEW_FORMAT_UINT_1X8',
    'HIP_RES_VIEW_FORMAT_UINT_2X16', 'HIP_RES_VIEW_FORMAT_UINT_2X32',
    'HIP_RES_VIEW_FORMAT_UINT_2X8', 'HIP_RES_VIEW_FORMAT_UINT_4X16',
    'HIP_RES_VIEW_FORMAT_UINT_4X32', 'HIP_RES_VIEW_FORMAT_UINT_4X8',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC1',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC2',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC3',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC4',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC5',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    'HIP_RES_VIEW_FORMAT_UNSIGNED_BC7', 'HIP_SUCCESS',
    'HIP_TEXTURE_DESC', 'HIP_TR_ADDRESS_MODE_BORDER',
    'HIP_TR_ADDRESS_MODE_CLAMP', 'HIP_TR_ADDRESS_MODE_MIRROR',
    'HIP_TR_ADDRESS_MODE_WRAP', 'HIP_TR_FILTER_MODE_LINEAR',
    'HIP_TR_FILTER_MODE_POINT', 'HIPaddress_mode',
    'HIPaddress_mode__enumvalues', 'HIPaddress_mode_enum',
    'HIPfilter_mode', 'HIPfilter_mode__enumvalues',
    'HIPfilter_mode_enum', 'HIPresourceViewFormat',
    'HIPresourceViewFormat__enumvalues', 'HIPresourceViewFormat_enum',
    'HIPresourcetype', 'HIPresourcetype__enumvalues',
    'HIPresourcetype_enum', '__hipGetPCH',
    '__hipPopCallConfiguration', '__hipPushCallConfiguration',
    'c__Ea_HIP_SUCCESS', 'dim3', 'hipAccessPolicyWindow',
    'hipAccessProperty', 'hipAccessPropertyNormal',
    'hipAccessPropertyPersisting', 'hipAccessPropertyStreaming',
    'hipAddressModeBorder', 'hipAddressModeClamp',
    'hipAddressModeMirror', 'hipAddressModeWrap', 'hipApiName',
    'hipArray3DCreate', 'hipArray3DGetDescriptor', 'hipArrayCreate',
    'hipArrayDestroy', 'hipArrayGetDescriptor', 'hipArrayGetInfo',
    'hipArrayMapInfo', 'hipArraySparseSubresourceType',
    'hipArraySparseSubresourceTypeMiptail',
    'hipArraySparseSubresourceTypeSparseLevel', 'hipArray_Format',
    'hipArray_const_t', 'hipArray_t', 'hipBindTexture',
    'hipBindTexture2D', 'hipBindTextureToArray',
    'hipBindTextureToMipmappedArray', 'hipChannelFormatDesc',
    'hipChannelFormatKind', 'hipChannelFormatKindFloat',
    'hipChannelFormatKindNone', 'hipChannelFormatKindSigned',
    'hipChannelFormatKindUnsigned', 'hipChooseDeviceR0600',
    'hipComputeMode', 'hipComputeModeDefault',
    'hipComputeModeExclusive', 'hipComputeModeExclusiveProcess',
    'hipComputeModeProhibited', 'hipConfigureCall',
    'hipCreateSurfaceObject', 'hipCreateTextureObject',
    'hipCtxCreate', 'hipCtxDestroy', 'hipCtxDisablePeerAccess',
    'hipCtxEnablePeerAccess', 'hipCtxGetApiVersion',
    'hipCtxGetCacheConfig', 'hipCtxGetCurrent', 'hipCtxGetDevice',
    'hipCtxGetFlags', 'hipCtxGetSharedMemConfig', 'hipCtxPopCurrent',
    'hipCtxPushCurrent', 'hipCtxSetCacheConfig', 'hipCtxSetCurrent',
    'hipCtxSetSharedMemConfig', 'hipCtxSynchronize', 'hipCtx_t',
    'hipDestroyExternalMemory', 'hipDestroyExternalSemaphore',
    'hipDestroySurfaceObject', 'hipDestroyTextureObject',
    'hipDevP2PAttrAccessSupported',
    'hipDevP2PAttrHipArrayAccessSupported',
    'hipDevP2PAttrNativeAtomicSupported',
    'hipDevP2PAttrPerformanceRank', 'hipDeviceArch_t',
    'hipDeviceAttributeAccessPolicyMaxWindowSize',
    'hipDeviceAttributeAmdSpecificBegin',
    'hipDeviceAttributeAmdSpecificEnd',
    'hipDeviceAttributeAsicRevision',
    'hipDeviceAttributeAsyncEngineCount',
    'hipDeviceAttributeCanMapHostMemory',
    'hipDeviceAttributeCanUseHostPointerForRegisteredMem',
    'hipDeviceAttributeCanUseStreamWaitValue',
    'hipDeviceAttributeClockInstructionRate',
    'hipDeviceAttributeClockRate',
    'hipDeviceAttributeComputeCapabilityMajor',
    'hipDeviceAttributeComputeCapabilityMinor',
    'hipDeviceAttributeComputeMode',
    'hipDeviceAttributeComputePreemptionSupported',
    'hipDeviceAttributeConcurrentKernels',
    'hipDeviceAttributeConcurrentManagedAccess',
    'hipDeviceAttributeCooperativeLaunch',
    'hipDeviceAttributeCooperativeMultiDeviceLaunch',
    'hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim',
    'hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc',
    'hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim',
    'hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem',
    'hipDeviceAttributeCudaCompatibleBegin',
    'hipDeviceAttributeCudaCompatibleEnd',
    'hipDeviceAttributeDeviceOverlap',
    'hipDeviceAttributeDirectManagedMemAccessFromHost',
    'hipDeviceAttributeEccEnabled',
    'hipDeviceAttributeFineGrainSupport',
    'hipDeviceAttributeGlobalL1CacheSupported',
    'hipDeviceAttributeHdpMemFlushCntl',
    'hipDeviceAttributeHdpRegFlushCntl',
    'hipDeviceAttributeHostNativeAtomicSupported',
    'hipDeviceAttributeHostRegisterSupported',
    'hipDeviceAttributeImageSupport', 'hipDeviceAttributeIntegrated',
    'hipDeviceAttributeIsLargeBar',
    'hipDeviceAttributeIsMultiGpuBoard',
    'hipDeviceAttributeKernelExecTimeout',
    'hipDeviceAttributeL2CacheSize',
    'hipDeviceAttributeLocalL1CacheSupported',
    'hipDeviceAttributeLuid', 'hipDeviceAttributeLuidDeviceNodeMask',
    'hipDeviceAttributeManagedMemory',
    'hipDeviceAttributeMaxBlockDimX',
    'hipDeviceAttributeMaxBlockDimY',
    'hipDeviceAttributeMaxBlockDimZ',
    'hipDeviceAttributeMaxBlocksPerMultiProcessor',
    'hipDeviceAttributeMaxGridDimX', 'hipDeviceAttributeMaxGridDimY',
    'hipDeviceAttributeMaxGridDimZ', 'hipDeviceAttributeMaxPitch',
    'hipDeviceAttributeMaxRegistersPerBlock',
    'hipDeviceAttributeMaxRegistersPerMultiprocessor',
    'hipDeviceAttributeMaxSharedMemoryPerBlock',
    'hipDeviceAttributeMaxSharedMemoryPerMultiprocessor',
    'hipDeviceAttributeMaxSurface1D',
    'hipDeviceAttributeMaxSurface1DLayered',
    'hipDeviceAttributeMaxSurface2D',
    'hipDeviceAttributeMaxSurface2DLayered',
    'hipDeviceAttributeMaxSurface3D',
    'hipDeviceAttributeMaxSurfaceCubemap',
    'hipDeviceAttributeMaxSurfaceCubemapLayered',
    'hipDeviceAttributeMaxTexture1DLayered',
    'hipDeviceAttributeMaxTexture1DLinear',
    'hipDeviceAttributeMaxTexture1DMipmap',
    'hipDeviceAttributeMaxTexture1DWidth',
    'hipDeviceAttributeMaxTexture2DGather',
    'hipDeviceAttributeMaxTexture2DHeight',
    'hipDeviceAttributeMaxTexture2DLayered',
    'hipDeviceAttributeMaxTexture2DLinear',
    'hipDeviceAttributeMaxTexture2DMipmap',
    'hipDeviceAttributeMaxTexture2DWidth',
    'hipDeviceAttributeMaxTexture3DAlt',
    'hipDeviceAttributeMaxTexture3DDepth',
    'hipDeviceAttributeMaxTexture3DHeight',
    'hipDeviceAttributeMaxTexture3DWidth',
    'hipDeviceAttributeMaxTextureCubemap',
    'hipDeviceAttributeMaxTextureCubemapLayered',
    'hipDeviceAttributeMaxThreadsDim',
    'hipDeviceAttributeMaxThreadsPerBlock',
    'hipDeviceAttributeMaxThreadsPerMultiProcessor',
    'hipDeviceAttributeMemoryBusWidth',
    'hipDeviceAttributeMemoryClockRate',
    'hipDeviceAttributeMemoryPoolsSupported',
    'hipDeviceAttributeMultiGpuBoardGroupID',
    'hipDeviceAttributeMultiprocessorCount',
    'hipDeviceAttributePageableMemoryAccess',
    'hipDeviceAttributePageableMemoryAccessUsesHostPageTables',
    'hipDeviceAttributePciBusId', 'hipDeviceAttributePciDeviceId',
    'hipDeviceAttributePciDomainID',
    'hipDeviceAttributePersistingL2CacheMaxSize',
    'hipDeviceAttributePhysicalMultiProcessorCount',
    'hipDeviceAttributeReservedSharedMemPerBlock',
    'hipDeviceAttributeSharedMemPerBlockOptin',
    'hipDeviceAttributeSharedMemPerMultiprocessor',
    'hipDeviceAttributeSingleToDoublePrecisionPerfRatio',
    'hipDeviceAttributeStreamPrioritiesSupported',
    'hipDeviceAttributeSurfaceAlignment',
    'hipDeviceAttributeTccDriver',
    'hipDeviceAttributeTextureAlignment',
    'hipDeviceAttributeTexturePitchAlignment',
    'hipDeviceAttributeTotalConstantMemory',
    'hipDeviceAttributeTotalGlobalMem',
    'hipDeviceAttributeUnifiedAddressing',
    'hipDeviceAttributeUnused1', 'hipDeviceAttributeUnused2',
    'hipDeviceAttributeUnused3', 'hipDeviceAttributeUnused4',
    'hipDeviceAttributeUnused5',
    'hipDeviceAttributeVendorSpecificBegin',
    'hipDeviceAttributeVirtualMemoryManagementSupported',
    'hipDeviceAttributeWallClockRate', 'hipDeviceAttributeWarpSize',
    'hipDeviceAttribute_t', 'hipDeviceCanAccessPeer',
    'hipDeviceComputeCapability', 'hipDeviceDisablePeerAccess',
    'hipDeviceEnablePeerAccess', 'hipDeviceGet',
    'hipDeviceGetAttribute', 'hipDeviceGetByPCIBusId',
    'hipDeviceGetCacheConfig', 'hipDeviceGetDefaultMemPool',
    'hipDeviceGetGraphMemAttribute', 'hipDeviceGetLimit',
    'hipDeviceGetMemPool', 'hipDeviceGetName',
    'hipDeviceGetP2PAttribute', 'hipDeviceGetPCIBusId',
    'hipDeviceGetSharedMemConfig', 'hipDeviceGetStreamPriorityRange',
    'hipDeviceGetUuid', 'hipDeviceGraphMemTrim', 'hipDeviceP2PAttr',
    'hipDevicePrimaryCtxGetState', 'hipDevicePrimaryCtxRelease',
    'hipDevicePrimaryCtxReset', 'hipDevicePrimaryCtxRetain',
    'hipDevicePrimaryCtxSetFlags', 'hipDeviceProp_tR0600',
    'hipDeviceReset', 'hipDeviceSetCacheConfig',
    'hipDeviceSetGraphMemAttribute', 'hipDeviceSetLimit',
    'hipDeviceSetMemPool', 'hipDeviceSetSharedMemConfig',
    'hipDeviceSynchronize', 'hipDeviceTotalMem', 'hipDevice_t',
    'hipDeviceptr_t', 'hipDriverGetVersion', 'hipDrvGetErrorName',
    'hipDrvGetErrorString', 'hipDrvGraphAddMemcpyNode',
    'hipDrvMemcpy2DUnaligned', 'hipDrvMemcpy3D',
    'hipDrvMemcpy3DAsync', 'hipDrvPointerGetAttributes',
    'hipErrorAlreadyAcquired', 'hipErrorAlreadyMapped',
    'hipErrorArrayIsMapped', 'hipErrorAssert',
    'hipErrorCapturedEvent', 'hipErrorContextAlreadyCurrent',
    'hipErrorContextAlreadyInUse', 'hipErrorContextIsDestroyed',
    'hipErrorCooperativeLaunchTooLarge', 'hipErrorDeinitialized',
    'hipErrorECCNotCorrectable', 'hipErrorFileNotFound',
    'hipErrorGraphExecUpdateFailure',
    'hipErrorHostMemoryAlreadyRegistered',
    'hipErrorHostMemoryNotRegistered', 'hipErrorIllegalAddress',
    'hipErrorIllegalState', 'hipErrorInitializationError',
    'hipErrorInsufficientDriver', 'hipErrorInvalidConfiguration',
    'hipErrorInvalidContext', 'hipErrorInvalidDevice',
    'hipErrorInvalidDeviceFunction', 'hipErrorInvalidDevicePointer',
    'hipErrorInvalidGraphicsContext', 'hipErrorInvalidHandle',
    'hipErrorInvalidImage', 'hipErrorInvalidKernelFile',
    'hipErrorInvalidMemcpyDirection', 'hipErrorInvalidPitchValue',
    'hipErrorInvalidResourceHandle', 'hipErrorInvalidSource',
    'hipErrorInvalidSymbol', 'hipErrorInvalidValue',
    'hipErrorLaunchFailure', 'hipErrorLaunchOutOfResources',
    'hipErrorLaunchTimeOut', 'hipErrorMapBufferObjectFailed',
    'hipErrorMapFailed', 'hipErrorMemoryAllocation',
    'hipErrorMissingConfiguration', 'hipErrorNoBinaryForGpu',
    'hipErrorNoDevice', 'hipErrorNotFound', 'hipErrorNotInitialized',
    'hipErrorNotMapped', 'hipErrorNotMappedAsArray',
    'hipErrorNotMappedAsPointer', 'hipErrorNotReady',
    'hipErrorNotSupported', 'hipErrorOperatingSystem',
    'hipErrorOutOfMemory', 'hipErrorPeerAccessAlreadyEnabled',
    'hipErrorPeerAccessNotEnabled', 'hipErrorPeerAccessUnsupported',
    'hipErrorPriorLaunchFailure', 'hipErrorProfilerAlreadyStarted',
    'hipErrorProfilerAlreadyStopped', 'hipErrorProfilerDisabled',
    'hipErrorProfilerNotInitialized', 'hipErrorRuntimeMemory',
    'hipErrorRuntimeOther', 'hipErrorSetOnActiveProcess',
    'hipErrorSharedObjectInitFailed',
    'hipErrorSharedObjectSymbolNotFound',
    'hipErrorStreamCaptureImplicit',
    'hipErrorStreamCaptureInvalidated',
    'hipErrorStreamCaptureIsolation', 'hipErrorStreamCaptureMerge',
    'hipErrorStreamCaptureUnjoined', 'hipErrorStreamCaptureUnmatched',
    'hipErrorStreamCaptureUnsupported',
    'hipErrorStreamCaptureWrongThread', 'hipErrorTbd',
    'hipErrorUnknown', 'hipErrorUnmapFailed',
    'hipErrorUnsupportedLimit', 'hipError_t', 'hipEventCreate',
    'hipEventCreateWithFlags', 'hipEventDestroy',
    'hipEventElapsedTime', 'hipEventQuery', 'hipEventRecord',
    'hipEventSynchronize', 'hipEvent_t', 'hipExtGetLastError',
    'hipExtGetLinkTypeAndHopCount', 'hipExtLaunchKernel',
    'hipExtLaunchMultiKernelMultiDevice', 'hipExtMallocWithFlags',
    'hipExtModuleLaunchKernel', 'hipExtStreamCreateWithCUMask',
    'hipExtStreamGetCUMask', 'hipExtent',
    'hipExternalMemoryBufferDesc', 'hipExternalMemoryGetMappedBuffer',
    'hipExternalMemoryGetMappedMipmappedArray',
    'hipExternalMemoryHandleDesc', 'hipExternalMemoryHandleType',
    'hipExternalMemoryHandleTypeD3D11Resource',
    'hipExternalMemoryHandleTypeD3D11ResourceKmt',
    'hipExternalMemoryHandleTypeD3D12Heap',
    'hipExternalMemoryHandleTypeD3D12Resource',
    'hipExternalMemoryHandleTypeNvSciBuf',
    'hipExternalMemoryHandleTypeOpaqueFd',
    'hipExternalMemoryHandleTypeOpaqueWin32',
    'hipExternalMemoryHandleTypeOpaqueWin32Kmt',
    'hipExternalMemoryHandleType__enumvalues',
    'hipExternalMemoryHandleType_enum',
    'hipExternalMemoryMipmappedArrayDesc', 'hipExternalMemory_t',
    'hipExternalSemaphoreHandleDesc',
    'hipExternalSemaphoreHandleType',
    'hipExternalSemaphoreHandleTypeD3D11Fence',
    'hipExternalSemaphoreHandleTypeD3D12Fence',
    'hipExternalSemaphoreHandleTypeKeyedMutex',
    'hipExternalSemaphoreHandleTypeKeyedMutexKmt',
    'hipExternalSemaphoreHandleTypeNvSciSync',
    'hipExternalSemaphoreHandleTypeOpaqueFd',
    'hipExternalSemaphoreHandleTypeOpaqueWin32',
    'hipExternalSemaphoreHandleTypeOpaqueWin32Kmt',
    'hipExternalSemaphoreHandleTypeTimelineSemaphoreFd',
    'hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32',
    'hipExternalSemaphoreHandleType__enumvalues',
    'hipExternalSemaphoreHandleType_enum',
    'hipExternalSemaphoreSignalNodeParams',
    'hipExternalSemaphoreSignalParams',
    'hipExternalSemaphoreWaitNodeParams',
    'hipExternalSemaphoreWaitParams', 'hipExternalSemaphore_t',
    'hipFilterModeLinear', 'hipFilterModePoint', 'hipFree',
    'hipFreeArray', 'hipFreeAsync', 'hipFreeHost',
    'hipFreeMipmappedArray', 'hipFuncAttribute',
    'hipFuncAttributeMax',
    'hipFuncAttributeMaxDynamicSharedMemorySize',
    'hipFuncAttributePreferredSharedMemoryCarveout',
    'hipFuncAttributes', 'hipFuncCachePreferEqual',
    'hipFuncCachePreferL1', 'hipFuncCachePreferNone',
    'hipFuncCachePreferShared', 'hipFuncCache_t',
    'hipFuncGetAttribute', 'hipFuncGetAttributes',
    'hipFuncSetAttribute', 'hipFuncSetCacheConfig',
    'hipFuncSetSharedMemConfig', 'hipFunctionLaunchParams',
    'hipFunction_attribute', 'hipFunction_t', 'hipGetChannelDesc',
    'hipGetDevice', 'hipGetDeviceCount', 'hipGetDeviceFlags',
    'hipGetDevicePropertiesR0600', 'hipGetErrorName',
    'hipGetErrorString', 'hipGetLastError',
    'hipGetMipmappedArrayLevel', 'hipGetStreamDeviceId',
    'hipGetSymbolAddress', 'hipGetSymbolSize',
    'hipGetTextureAlignmentOffset', 'hipGetTextureObjectResourceDesc',
    'hipGetTextureObjectResourceViewDesc',
    'hipGetTextureObjectTextureDesc', 'hipGetTextureReference',
    'hipGraphAddChildGraphNode', 'hipGraphAddDependencies',
    'hipGraphAddEmptyNode', 'hipGraphAddEventRecordNode',
    'hipGraphAddEventWaitNode', 'hipGraphAddHostNode',
    'hipGraphAddKernelNode', 'hipGraphAddMemAllocNode',
    'hipGraphAddMemFreeNode', 'hipGraphAddMemcpyNode',
    'hipGraphAddMemcpyNode1D', 'hipGraphAddMemcpyNodeFromSymbol',
    'hipGraphAddMemcpyNodeToSymbol', 'hipGraphAddMemsetNode',
    'hipGraphChildGraphNodeGetGraph', 'hipGraphClone',
    'hipGraphCreate', 'hipGraphDebugDotFlags',
    'hipGraphDebugDotFlagsEventNodeParams',
    'hipGraphDebugDotFlagsExtSemasSignalNodeParams',
    'hipGraphDebugDotFlagsExtSemasWaitNodeParams',
    'hipGraphDebugDotFlagsHandles',
    'hipGraphDebugDotFlagsHostNodeParams',
    'hipGraphDebugDotFlagsKernelNodeAttributes',
    'hipGraphDebugDotFlagsKernelNodeParams',
    'hipGraphDebugDotFlagsMemcpyNodeParams',
    'hipGraphDebugDotFlagsMemsetNodeParams',
    'hipGraphDebugDotFlagsVerbose', 'hipGraphDebugDotPrint',
    'hipGraphDestroy', 'hipGraphDestroyNode',
    'hipGraphEventRecordNodeGetEvent',
    'hipGraphEventRecordNodeSetEvent',
    'hipGraphEventWaitNodeGetEvent', 'hipGraphEventWaitNodeSetEvent',
    'hipGraphExecChildGraphNodeSetParams', 'hipGraphExecDestroy',
    'hipGraphExecEventRecordNodeSetEvent',
    'hipGraphExecEventWaitNodeSetEvent',
    'hipGraphExecHostNodeSetParams',
    'hipGraphExecKernelNodeSetParams',
    'hipGraphExecMemcpyNodeSetParams',
    'hipGraphExecMemcpyNodeSetParams1D',
    'hipGraphExecMemcpyNodeSetParamsFromSymbol',
    'hipGraphExecMemcpyNodeSetParamsToSymbol',
    'hipGraphExecMemsetNodeSetParams', 'hipGraphExecUpdate',
    'hipGraphExecUpdateError',
    'hipGraphExecUpdateErrorFunctionChanged',
    'hipGraphExecUpdateErrorNodeTypeChanged',
    'hipGraphExecUpdateErrorNotSupported',
    'hipGraphExecUpdateErrorParametersChanged',
    'hipGraphExecUpdateErrorTopologyChanged',
    'hipGraphExecUpdateErrorUnsupportedFunctionChange',
    'hipGraphExecUpdateResult', 'hipGraphExecUpdateSuccess',
    'hipGraphExec_t', 'hipGraphGetEdges', 'hipGraphGetNodes',
    'hipGraphGetRootNodes', 'hipGraphHostNodeGetParams',
    'hipGraphHostNodeSetParams', 'hipGraphInstantiate',
    'hipGraphInstantiateFlagAutoFreeOnLaunch',
    'hipGraphInstantiateFlagDeviceLaunch',
    'hipGraphInstantiateFlagUpload',
    'hipGraphInstantiateFlagUseNodePriority',
    'hipGraphInstantiateFlags', 'hipGraphInstantiateWithFlags',
    'hipGraphKernelNodeCopyAttributes',
    'hipGraphKernelNodeGetAttribute', 'hipGraphKernelNodeGetParams',
    'hipGraphKernelNodeSetAttribute', 'hipGraphKernelNodeSetParams',
    'hipGraphLaunch', 'hipGraphMemAllocNodeGetParams',
    'hipGraphMemAttrReservedMemCurrent',
    'hipGraphMemAttrReservedMemHigh', 'hipGraphMemAttrUsedMemCurrent',
    'hipGraphMemAttrUsedMemHigh', 'hipGraphMemAttributeType',
    'hipGraphMemFreeNodeGetParams', 'hipGraphMemcpyNodeGetParams',
    'hipGraphMemcpyNodeSetParams', 'hipGraphMemcpyNodeSetParams1D',
    'hipGraphMemcpyNodeSetParamsFromSymbol',
    'hipGraphMemcpyNodeSetParamsToSymbol',
    'hipGraphMemsetNodeGetParams', 'hipGraphMemsetNodeSetParams',
    'hipGraphNodeFindInClone', 'hipGraphNodeGetDependencies',
    'hipGraphNodeGetDependentNodes', 'hipGraphNodeGetEnabled',
    'hipGraphNodeGetType', 'hipGraphNodeSetEnabled',
    'hipGraphNodeType', 'hipGraphNodeTypeCount',
    'hipGraphNodeTypeEmpty', 'hipGraphNodeTypeEventRecord',
    'hipGraphNodeTypeExtSemaphoreSignal',
    'hipGraphNodeTypeExtSemaphoreWait', 'hipGraphNodeTypeGraph',
    'hipGraphNodeTypeHost', 'hipGraphNodeTypeKernel',
    'hipGraphNodeTypeMemAlloc', 'hipGraphNodeTypeMemFree',
    'hipGraphNodeTypeMemcpy', 'hipGraphNodeTypeMemcpyFromSymbol',
    'hipGraphNodeTypeMemcpyToSymbol', 'hipGraphNodeTypeMemset',
    'hipGraphNodeTypeWaitEvent', 'hipGraphNode_t',
    'hipGraphReleaseUserObject', 'hipGraphRemoveDependencies',
    'hipGraphRetainUserObject', 'hipGraphUpload',
    'hipGraphUserObjectMove', 'hipGraph_t', 'hipGraphicsMapResources',
    'hipGraphicsRegisterFlags', 'hipGraphicsRegisterFlagsNone',
    'hipGraphicsRegisterFlagsReadOnly',
    'hipGraphicsRegisterFlagsSurfaceLoadStore',
    'hipGraphicsRegisterFlagsTextureGather',
    'hipGraphicsRegisterFlagsWriteDiscard', 'hipGraphicsResource',
    'hipGraphicsResourceGetMappedPointer', 'hipGraphicsResource_t',
    'hipGraphicsSubResourceGetMappedArray',
    'hipGraphicsUnmapResources', 'hipGraphicsUnregisterResource',
    'hipHccModuleLaunchKernel', 'hipHostAlloc', 'hipHostFn_t',
    'hipHostFree', 'hipHostGetDevicePointer', 'hipHostGetFlags',
    'hipHostMalloc', 'hipHostNodeParams', 'hipHostRegister',
    'hipHostUnregister', 'hipImportExternalMemory',
    'hipImportExternalSemaphore', 'hipInit', 'hipIpcCloseMemHandle',
    'hipIpcEventHandle_t', 'hipIpcGetEventHandle',
    'hipIpcGetMemHandle', 'hipIpcMemHandle_t',
    'hipIpcOpenEventHandle', 'hipIpcOpenMemHandle', 'hipJitOption',
    'hipJitOptionCacheMode', 'hipJitOptionErrorLogBuffer',
    'hipJitOptionErrorLogBufferSizeBytes',
    'hipJitOptionFallbackStrategy', 'hipJitOptionFastCompile',
    'hipJitOptionGenerateDebugInfo', 'hipJitOptionGenerateLineInfo',
    'hipJitOptionInfoLogBuffer', 'hipJitOptionInfoLogBufferSizeBytes',
    'hipJitOptionLogVerbose', 'hipJitOptionMaxRegisters',
    'hipJitOptionNumOptions', 'hipJitOptionOptimizationLevel',
    'hipJitOptionSm3xOpt', 'hipJitOptionTarget',
    'hipJitOptionTargetFromContext', 'hipJitOptionThreadsPerBlock',
    'hipJitOptionWallTime', 'hipKernelNameRef',
    'hipKernelNameRefByPtr', 'hipKernelNodeAttrID',
    'hipKernelNodeAttrValue',
    'hipKernelNodeAttributeAccessPolicyWindow',
    'hipKernelNodeAttributeCooperative', 'hipKernelNodeParams',
    'hipLaunchByPtr', 'hipLaunchCooperativeKernel',
    'hipLaunchCooperativeKernelMultiDevice', 'hipLaunchHostFunc',
    'hipLaunchKernel', 'hipLaunchParams', 'hipLimitMallocHeapSize',
    'hipLimitPrintfFifoSize', 'hipLimitRange', 'hipLimitStackSize',
    'hipLimit_t', 'hipMalloc', 'hipMalloc3D', 'hipMalloc3DArray',
    'hipMallocArray', 'hipMallocAsync', 'hipMallocFromPoolAsync',
    'hipMallocHost', 'hipMallocManaged', 'hipMallocMipmappedArray',
    'hipMallocPitch', 'hipMemAccessDesc', 'hipMemAccessFlags',
    'hipMemAccessFlagsProtNone', 'hipMemAccessFlagsProtRead',
    'hipMemAccessFlagsProtReadWrite', 'hipMemAddressFree',
    'hipMemAddressReserve', 'hipMemAdvise',
    'hipMemAdviseSetAccessedBy', 'hipMemAdviseSetCoarseGrain',
    'hipMemAdviseSetPreferredLocation', 'hipMemAdviseSetReadMostly',
    'hipMemAdviseUnsetAccessedBy', 'hipMemAdviseUnsetCoarseGrain',
    'hipMemAdviseUnsetPreferredLocation',
    'hipMemAdviseUnsetReadMostly', 'hipMemAllocHost',
    'hipMemAllocNodeParams', 'hipMemAllocPitch',
    'hipMemAllocationGranularityMinimum',
    'hipMemAllocationGranularityRecommended',
    'hipMemAllocationGranularity_flags', 'hipMemAllocationHandleType',
    'hipMemAllocationProp', 'hipMemAllocationType',
    'hipMemAllocationTypeInvalid', 'hipMemAllocationTypeMax',
    'hipMemAllocationTypePinned', 'hipMemCreate',
    'hipMemExportToShareableHandle',
    'hipMemGenericAllocationHandle_t', 'hipMemGetAccess',
    'hipMemGetAddressRange', 'hipMemGetAllocationGranularity',
    'hipMemGetAllocationPropertiesFromHandle', 'hipMemGetInfo',
    'hipMemHandleType', 'hipMemHandleTypeGeneric',
    'hipMemHandleTypeNone', 'hipMemHandleTypePosixFileDescriptor',
    'hipMemHandleTypeWin32', 'hipMemHandleTypeWin32Kmt',
    'hipMemImportFromShareableHandle', 'hipMemLocation',
    'hipMemLocationType', 'hipMemLocationTypeDevice',
    'hipMemLocationTypeInvalid', 'hipMemMap', 'hipMemMapArrayAsync',
    'hipMemOperationType', 'hipMemOperationTypeMap',
    'hipMemOperationTypeUnmap', 'hipMemPoolAttr',
    'hipMemPoolAttrReleaseThreshold',
    'hipMemPoolAttrReservedMemCurrent',
    'hipMemPoolAttrReservedMemHigh', 'hipMemPoolAttrUsedMemCurrent',
    'hipMemPoolAttrUsedMemHigh', 'hipMemPoolCreate',
    'hipMemPoolDestroy', 'hipMemPoolExportPointer',
    'hipMemPoolExportToShareableHandle', 'hipMemPoolGetAccess',
    'hipMemPoolGetAttribute', 'hipMemPoolImportFromShareableHandle',
    'hipMemPoolImportPointer', 'hipMemPoolProps',
    'hipMemPoolPtrExportData',
    'hipMemPoolReuseAllowInternalDependencies',
    'hipMemPoolReuseAllowOpportunistic',
    'hipMemPoolReuseFollowEventDependencies', 'hipMemPoolSetAccess',
    'hipMemPoolSetAttribute', 'hipMemPoolTrimTo', 'hipMemPool_t',
    'hipMemPrefetchAsync', 'hipMemPtrGetInfo', 'hipMemRangeAttribute',
    'hipMemRangeAttributeAccessedBy',
    'hipMemRangeAttributeCoherencyMode',
    'hipMemRangeAttributeLastPrefetchLocation',
    'hipMemRangeAttributePreferredLocation',
    'hipMemRangeAttributeReadMostly', 'hipMemRangeCoherencyMode',
    'hipMemRangeCoherencyModeCoarseGrain',
    'hipMemRangeCoherencyModeFineGrain',
    'hipMemRangeCoherencyModeIndeterminate',
    'hipMemRangeGetAttribute', 'hipMemRangeGetAttributes',
    'hipMemRelease', 'hipMemRetainAllocationHandle',
    'hipMemSetAccess', 'hipMemUnmap', 'hipMemcpy', 'hipMemcpy2D',
    'hipMemcpy2DAsync', 'hipMemcpy2DFromArray',
    'hipMemcpy2DFromArrayAsync', 'hipMemcpy2DToArray',
    'hipMemcpy2DToArrayAsync', 'hipMemcpy3D', 'hipMemcpy3DAsync',
    'hipMemcpy3DParms', 'hipMemcpyAsync', 'hipMemcpyAtoH',
    'hipMemcpyDefault', 'hipMemcpyDeviceToDevice',
    'hipMemcpyDeviceToHost', 'hipMemcpyDtoD', 'hipMemcpyDtoDAsync',
    'hipMemcpyDtoH', 'hipMemcpyDtoHAsync', 'hipMemcpyFromArray',
    'hipMemcpyFromSymbol', 'hipMemcpyFromSymbolAsync',
    'hipMemcpyHostToDevice', 'hipMemcpyHostToHost', 'hipMemcpyHtoA',
    'hipMemcpyHtoD', 'hipMemcpyHtoDAsync', 'hipMemcpyKind',
    'hipMemcpyParam2D', 'hipMemcpyParam2DAsync', 'hipMemcpyPeer',
    'hipMemcpyPeerAsync', 'hipMemcpyToArray', 'hipMemcpyToSymbol',
    'hipMemcpyToSymbolAsync', 'hipMemcpyWithStream',
    'hipMemoryAdvise', 'hipMemoryType', 'hipMemoryTypeArray',
    'hipMemoryTypeDevice', 'hipMemoryTypeHost',
    'hipMemoryTypeManaged', 'hipMemoryTypeUnified',
    'hipMemoryTypeUnregistered', 'hipMemset', 'hipMemset2D',
    'hipMemset2DAsync', 'hipMemset3D', 'hipMemset3DAsync',
    'hipMemsetAsync', 'hipMemsetD16', 'hipMemsetD16Async',
    'hipMemsetD32', 'hipMemsetD32Async', 'hipMemsetD8',
    'hipMemsetD8Async', 'hipMemsetParams', 'hipMipmappedArray',
    'hipMipmappedArrayCreate', 'hipMipmappedArrayDestroy',
    'hipMipmappedArrayGetLevel', 'hipMipmappedArray_const_t',
    'hipMipmappedArray_t', 'hipModuleGetFunction',
    'hipModuleGetGlobal', 'hipModuleGetTexRef',
    'hipModuleLaunchCooperativeKernel',
    'hipModuleLaunchCooperativeKernelMultiDevice',
    'hipModuleLaunchKernel', 'hipModuleLoad', 'hipModuleLoadData',
    'hipModuleLoadDataEx',
    'hipModuleOccupancyMaxActiveBlocksPerMultiprocessor',
    'hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags',
    'hipModuleOccupancyMaxPotentialBlockSize',
    'hipModuleOccupancyMaxPotentialBlockSizeWithFlags',
    'hipModuleUnload', 'hipModule_t',
    'hipOccupancyMaxActiveBlocksPerMultiprocessor',
    'hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags',
    'hipOccupancyMaxPotentialBlockSize', 'hipPeekAtLastError',
    'hipPitchedPtr', 'hipPointerAttribute_t',
    'hipPointerGetAttribute', 'hipPointerGetAttributes',
    'hipPointerSetAttribute', 'hipPointer_attribute', 'hipPos',
    'hipProfilerStart', 'hipProfilerStop', 'hipReadModeElementType',
    'hipReadModeNormalizedFloat', 'hipResViewFormatFloat1',
    'hipResViewFormatFloat2', 'hipResViewFormatFloat4',
    'hipResViewFormatHalf1', 'hipResViewFormatHalf2',
    'hipResViewFormatHalf4', 'hipResViewFormatNone',
    'hipResViewFormatSignedBlockCompressed4',
    'hipResViewFormatSignedBlockCompressed5',
    'hipResViewFormatSignedBlockCompressed6H',
    'hipResViewFormatSignedChar1', 'hipResViewFormatSignedChar2',
    'hipResViewFormatSignedChar4', 'hipResViewFormatSignedInt1',
    'hipResViewFormatSignedInt2', 'hipResViewFormatSignedInt4',
    'hipResViewFormatSignedShort1', 'hipResViewFormatSignedShort2',
    'hipResViewFormatSignedShort4',
    'hipResViewFormatUnsignedBlockCompressed1',
    'hipResViewFormatUnsignedBlockCompressed2',
    'hipResViewFormatUnsignedBlockCompressed3',
    'hipResViewFormatUnsignedBlockCompressed4',
    'hipResViewFormatUnsignedBlockCompressed5',
    'hipResViewFormatUnsignedBlockCompressed6H',
    'hipResViewFormatUnsignedBlockCompressed7',
    'hipResViewFormatUnsignedChar1', 'hipResViewFormatUnsignedChar2',
    'hipResViewFormatUnsignedChar4', 'hipResViewFormatUnsignedInt1',
    'hipResViewFormatUnsignedInt2', 'hipResViewFormatUnsignedInt4',
    'hipResViewFormatUnsignedShort1',
    'hipResViewFormatUnsignedShort2',
    'hipResViewFormatUnsignedShort4', 'hipResourceDesc',
    'hipResourceType', 'hipResourceTypeArray',
    'hipResourceTypeLinear', 'hipResourceTypeMipmappedArray',
    'hipResourceTypePitch2D', 'hipResourceViewFormat',
    'hipResourcetype', 'hipResourcetype__enumvalues',
    'hipRuntimeGetVersion', 'hipSetDevice', 'hipSetDeviceFlags',
    'hipSetupArgument', 'hipSharedMemBankSizeDefault',
    'hipSharedMemBankSizeEightByte', 'hipSharedMemBankSizeFourByte',
    'hipSharedMemConfig', 'hipSignalExternalSemaphoresAsync',
    'hipStreamAddCallback', 'hipStreamAddCaptureDependencies',
    'hipStreamAttachMemAsync', 'hipStreamBeginCapture',
    'hipStreamCallback_t', 'hipStreamCaptureMode',
    'hipStreamCaptureModeGlobal', 'hipStreamCaptureModeRelaxed',
    'hipStreamCaptureModeThreadLocal', 'hipStreamCaptureStatus',
    'hipStreamCaptureStatusActive',
    'hipStreamCaptureStatusInvalidated', 'hipStreamCaptureStatusNone',
    'hipStreamCreate', 'hipStreamCreateWithFlags',
    'hipStreamCreateWithPriority', 'hipStreamDestroy',
    'hipStreamEndCapture', 'hipStreamGetCaptureInfo',
    'hipStreamGetCaptureInfo_v2', 'hipStreamGetDevice',
    'hipStreamGetFlags', 'hipStreamGetPriority',
    'hipStreamIsCapturing', 'hipStreamQuery',
    'hipStreamSetCaptureDependencies', 'hipStreamSynchronize',
    'hipStreamUpdateCaptureDependencies',
    'hipStreamUpdateCaptureDependenciesFlags', 'hipStreamWaitEvent',
    'hipStreamWaitValue32', 'hipStreamWaitValue64',
    'hipStreamWriteValue32', 'hipStreamWriteValue64', 'hipStream_t',
    'hipSuccess', 'hipSurfaceObject_t', 'hipTexObjectCreate',
    'hipTexObjectDestroy', 'hipTexObjectGetResourceDesc',
    'hipTexObjectGetResourceViewDesc', 'hipTexObjectGetTextureDesc',
    'hipTexRefGetAddress', 'hipTexRefGetAddressMode',
    'hipTexRefGetFilterMode', 'hipTexRefGetFlags',
    'hipTexRefGetFormat', 'hipTexRefGetMaxAnisotropy',
    'hipTexRefGetMipMappedArray', 'hipTexRefGetMipmapFilterMode',
    'hipTexRefGetMipmapLevelBias', 'hipTexRefGetMipmapLevelClamp',
    'hipTexRefSetAddress', 'hipTexRefSetAddress2D',
    'hipTexRefSetAddressMode', 'hipTexRefSetArray',
    'hipTexRefSetBorderColor', 'hipTexRefSetFilterMode',
    'hipTexRefSetFlags', 'hipTexRefSetFormat',
    'hipTexRefSetMaxAnisotropy', 'hipTexRefSetMipmapFilterMode',
    'hipTexRefSetMipmapLevelBias', 'hipTexRefSetMipmapLevelClamp',
    'hipTexRefSetMipmappedArray', 'hipTextureAddressMode',
    'hipTextureFilterMode', 'hipTextureObject_t',
    'hipTextureReadMode', 'hipThreadExchangeStreamCaptureMode',
    'hipUUID', 'hipUnbindTexture', 'hipUserObjectCreate',
    'hipUserObjectFlags', 'hipUserObjectNoDestructorSync',
    'hipUserObjectRelease', 'hipUserObjectRetain',
    'hipUserObjectRetainFlags', 'hipUserObject_t',
    'hipWaitExternalSemaphoresAsync', 'hip_Memcpy2D', 'hip_init',
    'hipmipmappedArray', 'hiprtcAddNameExpression',
    'hiprtcCompileProgram', 'hiprtcCreateProgram',
    'hiprtcDestroyProgram', 'hiprtcGetBitcode',
    'hiprtcGetBitcodeSize', 'hiprtcGetCode', 'hiprtcGetCodeSize',
    'hiprtcGetErrorString', 'hiprtcGetLoweredName',
    'hiprtcGetProgramLog', 'hiprtcGetProgramLogSize',
    'hiprtcJITInputType', 'hiprtcJIT_option', 'hiprtcLinkAddData',
    'hiprtcLinkAddFile', 'hiprtcLinkComplete', 'hiprtcLinkCreate',
    'hiprtcLinkDestroy', 'hiprtcLinkState', 'hiprtcProgram',
    'hiprtcResult', 'hiprtcVersion', 'make_hipExtent',
    'make_hipPitchedPtr', 'make_hipPos', 'size_t',
    'struct_HIP_ARRAY3D_DESCRIPTOR', 'struct_HIP_ARRAY_DESCRIPTOR',
    'struct_HIP_MEMCPY3D', 'struct_HIP_RESOURCE_DESC_st',
    'struct_HIP_RESOURCE_DESC_st_0_array',
    'struct_HIP_RESOURCE_DESC_st_0_linear',
    'struct_HIP_RESOURCE_DESC_st_0_mipmap',
    'struct_HIP_RESOURCE_DESC_st_0_pitch2D',
    'struct_HIP_RESOURCE_DESC_st_0_reserved',
    'struct_HIP_RESOURCE_VIEW_DESC_st', 'struct_HIP_TEXTURE_DESC_st',
    'struct___hip_surface', 'struct___hip_texture',
    'struct__hipGraphicsResource', 'struct__hiprtcProgram',
    'struct_c__SA_hipDeviceArch_t', 'struct_dim3',
    'struct_hipAccessPolicyWindow', 'struct_hipArray',
    'struct_hipArrayMapInfo', 'struct_hipArrayMapInfo_1_miptail',
    'struct_hipArrayMapInfo_1_sparseLevel',
    'struct_hipChannelFormatDesc', 'struct_hipDeviceProp_tR0600',
    'struct_hipExtent', 'struct_hipExternalMemoryBufferDesc_st',
    'struct_hipExternalMemoryHandleDesc_st',
    'struct_hipExternalMemoryHandleDesc_st_0_win32',
    'struct_hipExternalMemoryMipmappedArrayDesc_st',
    'struct_hipExternalSemaphoreHandleDesc_st',
    'struct_hipExternalSemaphoreHandleDesc_st_0_win32',
    'struct_hipExternalSemaphoreSignalNodeParams',
    'struct_hipExternalSemaphoreSignalParams_st',
    'struct_hipExternalSemaphoreSignalParams_st_0_fence',
    'struct_hipExternalSemaphoreSignalParams_st_0_keyedMutex',
    'struct_hipExternalSemaphoreSignalParams_st_params',
    'struct_hipExternalSemaphoreWaitNodeParams',
    'struct_hipExternalSemaphoreWaitParams_st',
    'struct_hipExternalSemaphoreWaitParams_st_0_fence',
    'struct_hipExternalSemaphoreWaitParams_st_0_keyedMutex',
    'struct_hipExternalSemaphoreWaitParams_st_params',
    'struct_hipFuncAttributes', 'struct_hipFunctionLaunchParams_t',
    'struct_hipGraphExec', 'struct_hipGraphNode',
    'struct_hipHostNodeParams', 'struct_hipIpcEventHandle_st',
    'struct_hipIpcMemHandle_st', 'struct_hipKernelNodeParams',
    'struct_hipLaunchParams_t', 'struct_hipMemAccessDesc',
    'struct_hipMemAllocNodeParams', 'struct_hipMemAllocationProp',
    'struct_hipMemAllocationProp_allocFlags', 'struct_hipMemLocation',
    'struct_hipMemPoolProps', 'struct_hipMemPoolPtrExportData',
    'struct_hipMemcpy3DParms', 'struct_hipMemsetParams',
    'struct_hipMipmappedArray', 'struct_hipPitchedPtr',
    'struct_hipPointerAttribute_t', 'struct_hipPos',
    'struct_hipResourceDesc', 'struct_hipResourceDesc_0_array',
    'struct_hipResourceDesc_0_linear',
    'struct_hipResourceDesc_0_mipmap',
    'struct_hipResourceDesc_0_pitch2D', 'struct_hipResourceViewDesc',
    'struct_hipTextureDesc', 'struct_hipUUID_t',
    'struct_hipUserObject', 'struct_hip_Memcpy2D', 'struct_ihipCtx_t',
    'struct_ihipEvent_t', 'struct_ihipGraph',
    'struct_ihipMemGenericAllocationHandle',
    'struct_ihipMemPoolHandle_t', 'struct_ihipModuleSymbol_t',
    'struct_ihipModule_t', 'struct_ihipStream_t',
    'struct_ihiprtcLinkState', 'struct_textureReference', 'uint32_t',
    'uint64_t', 'union_HIP_RESOURCE_DESC_st_res',
    'union_hipArrayMapInfo_memHandle',
    'union_hipArrayMapInfo_resource',
    'union_hipArrayMapInfo_subresource',
    'union_hipExternalMemoryHandleDesc_st_handle',
    'union_hipExternalSemaphoreHandleDesc_st_handle',
    'union_hipExternalSemaphoreSignalParams_st_0_nvSciSync',
    'union_hipExternalSemaphoreWaitParams_st_0_nvSciSync',
    'union_hipKernelNodeAttrValue', 'union_hipResourceDesc_res']
hipDeviceProp_t = hipDeviceProp_tR0600
hipGetDeviceProperties = hipGetDevicePropertiesR0600


# tinygrad/runtime/autogen/hsa.py

# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries = {}
_libraries['libhsa-runtime64.so'] = ctypes.CDLL('/opt/rocm/lib/libhsa-runtime64.so')
class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16




# values for enumeration 'hsa_status_t'
hsa_status_t__enumvalues = {
    0: 'HSA_STATUS_SUCCESS',
    1: 'HSA_STATUS_INFO_BREAK',
    4096: 'HSA_STATUS_ERROR',
    4097: 'HSA_STATUS_ERROR_INVALID_ARGUMENT',
    4098: 'HSA_STATUS_ERROR_INVALID_QUEUE_CREATION',
    4099: 'HSA_STATUS_ERROR_INVALID_ALLOCATION',
    4100: 'HSA_STATUS_ERROR_INVALID_AGENT',
    4101: 'HSA_STATUS_ERROR_INVALID_REGION',
    4102: 'HSA_STATUS_ERROR_INVALID_SIGNAL',
    4103: 'HSA_STATUS_ERROR_INVALID_QUEUE',
    4104: 'HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    4105: 'HSA_STATUS_ERROR_INVALID_PACKET_FORMAT',
    4106: 'HSA_STATUS_ERROR_RESOURCE_FREE',
    4107: 'HSA_STATUS_ERROR_NOT_INITIALIZED',
    4108: 'HSA_STATUS_ERROR_REFCOUNT_OVERFLOW',
    4109: 'HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS',
    4110: 'HSA_STATUS_ERROR_INVALID_INDEX',
    4111: 'HSA_STATUS_ERROR_INVALID_ISA',
    4119: 'HSA_STATUS_ERROR_INVALID_ISA_NAME',
    4112: 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT',
    4113: 'HSA_STATUS_ERROR_INVALID_EXECUTABLE',
    4114: 'HSA_STATUS_ERROR_FROZEN_EXECUTABLE',
    4115: 'HSA_STATUS_ERROR_INVALID_SYMBOL_NAME',
    4116: 'HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED',
    4117: 'HSA_STATUS_ERROR_VARIABLE_UNDEFINED',
    4118: 'HSA_STATUS_ERROR_EXCEPTION',
    4120: 'HSA_STATUS_ERROR_INVALID_CODE_SYMBOL',
    4121: 'HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL',
    4128: 'HSA_STATUS_ERROR_INVALID_FILE',
    4129: 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER',
    4130: 'HSA_STATUS_ERROR_INVALID_CACHE',
    4131: 'HSA_STATUS_ERROR_INVALID_WAVEFRONT',
    4132: 'HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP',
    4133: 'HSA_STATUS_ERROR_INVALID_RUNTIME_STATE',
    4134: 'HSA_STATUS_ERROR_FATAL',
}
HSA_STATUS_SUCCESS = 0
HSA_STATUS_INFO_BREAK = 1
HSA_STATUS_ERROR = 4096
HSA_STATUS_ERROR_INVALID_ARGUMENT = 4097
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 4098
HSA_STATUS_ERROR_INVALID_ALLOCATION = 4099
HSA_STATUS_ERROR_INVALID_AGENT = 4100
HSA_STATUS_ERROR_INVALID_REGION = 4101
HSA_STATUS_ERROR_INVALID_SIGNAL = 4102
HSA_STATUS_ERROR_INVALID_QUEUE = 4103
HSA_STATUS_ERROR_OUT_OF_RESOURCES = 4104
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 4105
HSA_STATUS_ERROR_RESOURCE_FREE = 4106
HSA_STATUS_ERROR_NOT_INITIALIZED = 4107
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 4108
HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 4109
HSA_STATUS_ERROR_INVALID_INDEX = 4110
HSA_STATUS_ERROR_INVALID_ISA = 4111
HSA_STATUS_ERROR_INVALID_ISA_NAME = 4119
HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 4112
HSA_STATUS_ERROR_INVALID_EXECUTABLE = 4113
HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 4114
HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 4115
HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 4116
HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 4117
HSA_STATUS_ERROR_EXCEPTION = 4118
HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 4120
HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 4121
HSA_STATUS_ERROR_INVALID_FILE = 4128
HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 4129
HSA_STATUS_ERROR_INVALID_CACHE = 4130
HSA_STATUS_ERROR_INVALID_WAVEFRONT = 4131
HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 4132
HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 4133
HSA_STATUS_ERROR_FATAL = 4134
hsa_status_t = ctypes.c_uint32 # enum
try:
    hsa_status_string = _libraries['libhsa-runtime64.so'].hsa_status_string
    hsa_status_string.restype = hsa_status_t
    hsa_status_string.argtypes = [hsa_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
class struct_hsa_dim3_s(Structure):
    pass

struct_hsa_dim3_s._pack_ = 1 # source:False
struct_hsa_dim3_s._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
]

hsa_dim3_t = struct_hsa_dim3_s

# values for enumeration 'hsa_access_permission_t'
hsa_access_permission_t__enumvalues = {
    0: 'HSA_ACCESS_PERMISSION_NONE',
    1: 'HSA_ACCESS_PERMISSION_RO',
    2: 'HSA_ACCESS_PERMISSION_WO',
    3: 'HSA_ACCESS_PERMISSION_RW',
}
HSA_ACCESS_PERMISSION_NONE = 0
HSA_ACCESS_PERMISSION_RO = 1
HSA_ACCESS_PERMISSION_WO = 2
HSA_ACCESS_PERMISSION_RW = 3
hsa_access_permission_t = ctypes.c_uint32 # enum
hsa_file_t = ctypes.c_int32
try:
    hsa_init = _libraries['libhsa-runtime64.so'].hsa_init
    hsa_init.restype = hsa_status_t
    hsa_init.argtypes = []
except AttributeError:
    pass
try:
    hsa_shut_down = _libraries['libhsa-runtime64.so'].hsa_shut_down
    hsa_shut_down.restype = hsa_status_t
    hsa_shut_down.argtypes = []
except AttributeError:
    pass

# values for enumeration 'hsa_endianness_t'
hsa_endianness_t__enumvalues = {
    0: 'HSA_ENDIANNESS_LITTLE',
    1: 'HSA_ENDIANNESS_BIG',
}
HSA_ENDIANNESS_LITTLE = 0
HSA_ENDIANNESS_BIG = 1
hsa_endianness_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_machine_model_t'
hsa_machine_model_t__enumvalues = {
    0: 'HSA_MACHINE_MODEL_SMALL',
    1: 'HSA_MACHINE_MODEL_LARGE',
}
HSA_MACHINE_MODEL_SMALL = 0
HSA_MACHINE_MODEL_LARGE = 1
hsa_machine_model_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_profile_t'
hsa_profile_t__enumvalues = {
    0: 'HSA_PROFILE_BASE',
    1: 'HSA_PROFILE_FULL',
}
HSA_PROFILE_BASE = 0
HSA_PROFILE_FULL = 1
hsa_profile_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_system_info_t'
hsa_system_info_t__enumvalues = {
    0: 'HSA_SYSTEM_INFO_VERSION_MAJOR',
    1: 'HSA_SYSTEM_INFO_VERSION_MINOR',
    2: 'HSA_SYSTEM_INFO_TIMESTAMP',
    3: 'HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY',
    4: 'HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT',
    5: 'HSA_SYSTEM_INFO_ENDIANNESS',
    6: 'HSA_SYSTEM_INFO_MACHINE_MODEL',
    7: 'HSA_SYSTEM_INFO_EXTENSIONS',
    512: 'HSA_AMD_SYSTEM_INFO_BUILD_VERSION',
    513: 'HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED',
    514: 'HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT',
    515: 'HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED',
    516: 'HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED',
    517: 'HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED',
    518: 'HSA_AMD_SYSTEM_INFO_XNACK_ENABLED',
}
HSA_SYSTEM_INFO_VERSION_MAJOR = 0
HSA_SYSTEM_INFO_VERSION_MINOR = 1
HSA_SYSTEM_INFO_TIMESTAMP = 2
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = 4
HSA_SYSTEM_INFO_ENDIANNESS = 5
HSA_SYSTEM_INFO_MACHINE_MODEL = 6
HSA_SYSTEM_INFO_EXTENSIONS = 7
HSA_AMD_SYSTEM_INFO_BUILD_VERSION = 512
HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = 513
HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = 514
HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED = 515
HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = 516
HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED = 517
HSA_AMD_SYSTEM_INFO_XNACK_ENABLED = 518
hsa_system_info_t = ctypes.c_uint32 # enum
try:
    hsa_system_get_info = _libraries['libhsa-runtime64.so'].hsa_system_get_info
    hsa_system_get_info.restype = hsa_status_t
    hsa_system_get_info.argtypes = [hsa_system_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_extension_t'
hsa_extension_t__enumvalues = {
    0: 'HSA_EXTENSION_FINALIZER',
    1: 'HSA_EXTENSION_IMAGES',
    2: 'HSA_EXTENSION_PERFORMANCE_COUNTERS',
    3: 'HSA_EXTENSION_PROFILING_EVENTS',
    3: 'HSA_EXTENSION_STD_LAST',
    512: 'HSA_AMD_FIRST_EXTENSION',
    512: 'HSA_EXTENSION_AMD_PROFILER',
    513: 'HSA_EXTENSION_AMD_LOADER',
    514: 'HSA_EXTENSION_AMD_AQLPROFILE',
    514: 'HSA_AMD_LAST_EXTENSION',
}
HSA_EXTENSION_FINALIZER = 0
HSA_EXTENSION_IMAGES = 1
HSA_EXTENSION_PERFORMANCE_COUNTERS = 2
HSA_EXTENSION_PROFILING_EVENTS = 3
HSA_EXTENSION_STD_LAST = 3
HSA_AMD_FIRST_EXTENSION = 512
HSA_EXTENSION_AMD_PROFILER = 512
HSA_EXTENSION_AMD_LOADER = 513
HSA_EXTENSION_AMD_AQLPROFILE = 514
HSA_AMD_LAST_EXTENSION = 514
hsa_extension_t = ctypes.c_uint32 # enum
uint16_t = ctypes.c_uint16
try:
    hsa_extension_get_name = _libraries['libhsa-runtime64.so'].hsa_extension_get_name
    hsa_extension_get_name.restype = hsa_status_t
    hsa_extension_get_name.argtypes = [uint16_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hsa_system_extension_supported = _libraries['libhsa-runtime64.so'].hsa_system_extension_supported
    hsa_system_extension_supported.restype = hsa_status_t
    hsa_system_extension_supported.argtypes = [uint16_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_system_major_extension_supported = _libraries['libhsa-runtime64.so'].hsa_system_major_extension_supported
    hsa_system_major_extension_supported.restype = hsa_status_t
    hsa_system_major_extension_supported.argtypes = [uint16_t, uint16_t, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_system_get_extension_table = _libraries['libhsa-runtime64.so'].hsa_system_get_extension_table
    hsa_system_get_extension_table.restype = hsa_status_t
    hsa_system_get_extension_table.argtypes = [uint16_t, uint16_t, uint16_t, ctypes.POINTER(None)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    hsa_system_get_major_extension_table = _libraries['libhsa-runtime64.so'].hsa_system_get_major_extension_table
    hsa_system_get_major_extension_table.restype = hsa_status_t
    hsa_system_get_major_extension_table.argtypes = [uint16_t, uint16_t, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_agent_s(Structure):
    pass

struct_hsa_agent_s._pack_ = 1 # source:False
struct_hsa_agent_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_agent_t = struct_hsa_agent_s

# values for enumeration 'hsa_agent_feature_t'
hsa_agent_feature_t__enumvalues = {
    1: 'HSA_AGENT_FEATURE_KERNEL_DISPATCH',
    2: 'HSA_AGENT_FEATURE_AGENT_DISPATCH',
}
HSA_AGENT_FEATURE_KERNEL_DISPATCH = 1
HSA_AGENT_FEATURE_AGENT_DISPATCH = 2
hsa_agent_feature_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_device_type_t'
hsa_device_type_t__enumvalues = {
    0: 'HSA_DEVICE_TYPE_CPU',
    1: 'HSA_DEVICE_TYPE_GPU',
    2: 'HSA_DEVICE_TYPE_DSP',
}
HSA_DEVICE_TYPE_CPU = 0
HSA_DEVICE_TYPE_GPU = 1
HSA_DEVICE_TYPE_DSP = 2
hsa_device_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_default_float_rounding_mode_t'
hsa_default_float_rounding_mode_t__enumvalues = {
    0: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT',
    1: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO',
    2: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR',
}
HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 0
HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = 1
HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = 2
hsa_default_float_rounding_mode_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_agent_info_t'
hsa_agent_info_t__enumvalues = {
    0: 'HSA_AGENT_INFO_NAME',
    1: 'HSA_AGENT_INFO_VENDOR_NAME',
    2: 'HSA_AGENT_INFO_FEATURE',
    3: 'HSA_AGENT_INFO_MACHINE_MODEL',
    4: 'HSA_AGENT_INFO_PROFILE',
    5: 'HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    23: 'HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    24: 'HSA_AGENT_INFO_FAST_F16_OPERATION',
    6: 'HSA_AGENT_INFO_WAVEFRONT_SIZE',
    7: 'HSA_AGENT_INFO_WORKGROUP_MAX_DIM',
    8: 'HSA_AGENT_INFO_WORKGROUP_MAX_SIZE',
    9: 'HSA_AGENT_INFO_GRID_MAX_DIM',
    10: 'HSA_AGENT_INFO_GRID_MAX_SIZE',
    11: 'HSA_AGENT_INFO_FBARRIER_MAX_SIZE',
    12: 'HSA_AGENT_INFO_QUEUES_MAX',
    13: 'HSA_AGENT_INFO_QUEUE_MIN_SIZE',
    14: 'HSA_AGENT_INFO_QUEUE_MAX_SIZE',
    15: 'HSA_AGENT_INFO_QUEUE_TYPE',
    16: 'HSA_AGENT_INFO_NODE',
    17: 'HSA_AGENT_INFO_DEVICE',
    18: 'HSA_AGENT_INFO_CACHE_SIZE',
    19: 'HSA_AGENT_INFO_ISA',
    20: 'HSA_AGENT_INFO_EXTENSIONS',
    21: 'HSA_AGENT_INFO_VERSION_MAJOR',
    22: 'HSA_AGENT_INFO_VERSION_MINOR',
    2147483647: 'HSA_AGENT_INFO_LAST',
}
HSA_AGENT_INFO_NAME = 0
HSA_AGENT_INFO_VENDOR_NAME = 1
HSA_AGENT_INFO_FEATURE = 2
HSA_AGENT_INFO_MACHINE_MODEL = 3
HSA_AGENT_INFO_PROFILE = 4
HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23
HSA_AGENT_INFO_FAST_F16_OPERATION = 24
HSA_AGENT_INFO_WAVEFRONT_SIZE = 6
HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8
HSA_AGENT_INFO_GRID_MAX_DIM = 9
HSA_AGENT_INFO_GRID_MAX_SIZE = 10
HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11
HSA_AGENT_INFO_QUEUES_MAX = 12
HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13
HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14
HSA_AGENT_INFO_QUEUE_TYPE = 15
HSA_AGENT_INFO_NODE = 16
HSA_AGENT_INFO_DEVICE = 17
HSA_AGENT_INFO_CACHE_SIZE = 18
HSA_AGENT_INFO_ISA = 19
HSA_AGENT_INFO_EXTENSIONS = 20
HSA_AGENT_INFO_VERSION_MAJOR = 21
HSA_AGENT_INFO_VERSION_MINOR = 22
HSA_AGENT_INFO_LAST = 2147483647
hsa_agent_info_t = ctypes.c_uint32 # enum
try:
    hsa_agent_get_info = _libraries['libhsa-runtime64.so'].hsa_agent_get_info
    hsa_agent_get_info.restype = hsa_status_t
    hsa_agent_get_info.argtypes = [hsa_agent_t, hsa_agent_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_iterate_agents = _libraries['libhsa-runtime64.so'].hsa_iterate_agents
    hsa_iterate_agents.restype = hsa_status_t
    hsa_iterate_agents.argtypes = [ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_exception_policy_t'
hsa_exception_policy_t__enumvalues = {
    1: 'HSA_EXCEPTION_POLICY_BREAK',
    2: 'HSA_EXCEPTION_POLICY_DETECT',
}
HSA_EXCEPTION_POLICY_BREAK = 1
HSA_EXCEPTION_POLICY_DETECT = 2
hsa_exception_policy_t = ctypes.c_uint32 # enum
try:
    hsa_agent_get_exception_policies = _libraries['libhsa-runtime64.so'].hsa_agent_get_exception_policies
    hsa_agent_get_exception_policies.restype = hsa_status_t
    hsa_agent_get_exception_policies.argtypes = [hsa_agent_t, hsa_profile_t, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
class struct_hsa_cache_s(Structure):
    pass

struct_hsa_cache_s._pack_ = 1 # source:False
struct_hsa_cache_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_cache_t = struct_hsa_cache_s

# values for enumeration 'hsa_cache_info_t'
hsa_cache_info_t__enumvalues = {
    0: 'HSA_CACHE_INFO_NAME_LENGTH',
    1: 'HSA_CACHE_INFO_NAME',
    2: 'HSA_CACHE_INFO_LEVEL',
    3: 'HSA_CACHE_INFO_SIZE',
}
HSA_CACHE_INFO_NAME_LENGTH = 0
HSA_CACHE_INFO_NAME = 1
HSA_CACHE_INFO_LEVEL = 2
HSA_CACHE_INFO_SIZE = 3
hsa_cache_info_t = ctypes.c_uint32 # enum
try:
    hsa_cache_get_info = _libraries['libhsa-runtime64.so'].hsa_cache_get_info
    hsa_cache_get_info.restype = hsa_status_t
    hsa_cache_get_info.argtypes = [hsa_cache_t, hsa_cache_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_caches = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_caches
    hsa_agent_iterate_caches.restype = hsa_status_t
    hsa_agent_iterate_caches.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_cache_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_extension_supported = _libraries['libhsa-runtime64.so'].hsa_agent_extension_supported
    hsa_agent_extension_supported.restype = hsa_status_t
    hsa_agent_extension_supported.argtypes = [uint16_t, hsa_agent_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_agent_major_extension_supported = _libraries['libhsa-runtime64.so'].hsa_agent_major_extension_supported
    hsa_agent_major_extension_supported.restype = hsa_status_t
    hsa_agent_major_extension_supported.argtypes = [uint16_t, hsa_agent_t, uint16_t, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
class struct_hsa_signal_s(Structure):
    pass

struct_hsa_signal_s._pack_ = 1 # source:False
struct_hsa_signal_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_signal_t = struct_hsa_signal_s
hsa_signal_value_t = ctypes.c_int64
uint32_t = ctypes.c_uint32
try:
    hsa_signal_create = _libraries['libhsa-runtime64.so'].hsa_signal_create
    hsa_signal_create.restype = hsa_status_t
    hsa_signal_create.argtypes = [hsa_signal_value_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass
try:
    hsa_signal_destroy = _libraries['libhsa-runtime64.so'].hsa_signal_destroy
    hsa_signal_destroy.restype = hsa_status_t
    hsa_signal_destroy.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_load_scacquire
    hsa_signal_load_scacquire.restype = hsa_signal_value_t
    hsa_signal_load_scacquire.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_load_relaxed
    hsa_signal_load_relaxed.restype = hsa_signal_value_t
    hsa_signal_load_relaxed.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_load_acquire
    hsa_signal_load_acquire.restype = hsa_signal_value_t
    hsa_signal_load_acquire.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_store_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_store_relaxed
    hsa_signal_store_relaxed.restype = None
    hsa_signal_store_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_store_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_store_screlease
    hsa_signal_store_screlease.restype = None
    hsa_signal_store_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_store_release = _libraries['libhsa-runtime64.so'].hsa_signal_store_release
    hsa_signal_store_release.restype = None
    hsa_signal_store_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_silent_store_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_silent_store_relaxed
    hsa_signal_silent_store_relaxed.restype = None
    hsa_signal_silent_store_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_silent_store_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_silent_store_screlease
    hsa_signal_silent_store_screlease.restype = None
    hsa_signal_silent_store_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_scacq_screl
    hsa_signal_exchange_scacq_screl.restype = hsa_signal_value_t
    hsa_signal_exchange_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_acq_rel
    hsa_signal_exchange_acq_rel.restype = hsa_signal_value_t
    hsa_signal_exchange_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_scacquire
    hsa_signal_exchange_scacquire.restype = hsa_signal_value_t
    hsa_signal_exchange_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_acquire
    hsa_signal_exchange_acquire.restype = hsa_signal_value_t
    hsa_signal_exchange_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_relaxed
    hsa_signal_exchange_relaxed.restype = hsa_signal_value_t
    hsa_signal_exchange_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_screlease
    hsa_signal_exchange_screlease.restype = hsa_signal_value_t
    hsa_signal_exchange_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_release = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_release
    hsa_signal_exchange_release.restype = hsa_signal_value_t
    hsa_signal_exchange_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_cas_scacq_screl
    hsa_signal_cas_scacq_screl.restype = hsa_signal_value_t
    hsa_signal_cas_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_cas_acq_rel
    hsa_signal_cas_acq_rel.restype = hsa_signal_value_t
    hsa_signal_cas_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_cas_scacquire
    hsa_signal_cas_scacquire.restype = hsa_signal_value_t
    hsa_signal_cas_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_cas_acquire
    hsa_signal_cas_acquire.restype = hsa_signal_value_t
    hsa_signal_cas_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_cas_relaxed
    hsa_signal_cas_relaxed.restype = hsa_signal_value_t
    hsa_signal_cas_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_cas_screlease
    hsa_signal_cas_screlease.restype = hsa_signal_value_t
    hsa_signal_cas_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_release = _libraries['libhsa-runtime64.so'].hsa_signal_cas_release
    hsa_signal_cas_release.restype = hsa_signal_value_t
    hsa_signal_cas_release.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_add_scacq_screl
    hsa_signal_add_scacq_screl.restype = None
    hsa_signal_add_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_add_acq_rel
    hsa_signal_add_acq_rel.restype = None
    hsa_signal_add_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_add_scacquire
    hsa_signal_add_scacquire.restype = None
    hsa_signal_add_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_add_acquire
    hsa_signal_add_acquire.restype = None
    hsa_signal_add_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_add_relaxed
    hsa_signal_add_relaxed.restype = None
    hsa_signal_add_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_add_screlease
    hsa_signal_add_screlease.restype = None
    hsa_signal_add_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_release = _libraries['libhsa-runtime64.so'].hsa_signal_add_release
    hsa_signal_add_release.restype = None
    hsa_signal_add_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_scacq_screl
    hsa_signal_subtract_scacq_screl.restype = None
    hsa_signal_subtract_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_acq_rel
    hsa_signal_subtract_acq_rel.restype = None
    hsa_signal_subtract_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_scacquire
    hsa_signal_subtract_scacquire.restype = None
    hsa_signal_subtract_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_acquire
    hsa_signal_subtract_acquire.restype = None
    hsa_signal_subtract_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_relaxed
    hsa_signal_subtract_relaxed.restype = None
    hsa_signal_subtract_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_screlease
    hsa_signal_subtract_screlease.restype = None
    hsa_signal_subtract_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_release = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_release
    hsa_signal_subtract_release.restype = None
    hsa_signal_subtract_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_and_scacq_screl
    hsa_signal_and_scacq_screl.restype = None
    hsa_signal_and_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_and_acq_rel
    hsa_signal_and_acq_rel.restype = None
    hsa_signal_and_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_and_scacquire
    hsa_signal_and_scacquire.restype = None
    hsa_signal_and_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_and_acquire
    hsa_signal_and_acquire.restype = None
    hsa_signal_and_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_and_relaxed
    hsa_signal_and_relaxed.restype = None
    hsa_signal_and_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_and_screlease
    hsa_signal_and_screlease.restype = None
    hsa_signal_and_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_release = _libraries['libhsa-runtime64.so'].hsa_signal_and_release
    hsa_signal_and_release.restype = None
    hsa_signal_and_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_or_scacq_screl
    hsa_signal_or_scacq_screl.restype = None
    hsa_signal_or_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_or_acq_rel
    hsa_signal_or_acq_rel.restype = None
    hsa_signal_or_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_or_scacquire
    hsa_signal_or_scacquire.restype = None
    hsa_signal_or_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_or_acquire
    hsa_signal_or_acquire.restype = None
    hsa_signal_or_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_or_relaxed
    hsa_signal_or_relaxed.restype = None
    hsa_signal_or_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_or_screlease
    hsa_signal_or_screlease.restype = None
    hsa_signal_or_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_release = _libraries['libhsa-runtime64.so'].hsa_signal_or_release
    hsa_signal_or_release.restype = None
    hsa_signal_or_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_xor_scacq_screl
    hsa_signal_xor_scacq_screl.restype = None
    hsa_signal_xor_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_xor_acq_rel
    hsa_signal_xor_acq_rel.restype = None
    hsa_signal_xor_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_xor_scacquire
    hsa_signal_xor_scacquire.restype = None
    hsa_signal_xor_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_xor_acquire
    hsa_signal_xor_acquire.restype = None
    hsa_signal_xor_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_xor_relaxed
    hsa_signal_xor_relaxed.restype = None
    hsa_signal_xor_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_xor_screlease
    hsa_signal_xor_screlease.restype = None
    hsa_signal_xor_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_release = _libraries['libhsa-runtime64.so'].hsa_signal_xor_release
    hsa_signal_xor_release.restype = None
    hsa_signal_xor_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass

# values for enumeration 'hsa_signal_condition_t'
hsa_signal_condition_t__enumvalues = {
    0: 'HSA_SIGNAL_CONDITION_EQ',
    1: 'HSA_SIGNAL_CONDITION_NE',
    2: 'HSA_SIGNAL_CONDITION_LT',
    3: 'HSA_SIGNAL_CONDITION_GTE',
}
HSA_SIGNAL_CONDITION_EQ = 0
HSA_SIGNAL_CONDITION_NE = 1
HSA_SIGNAL_CONDITION_LT = 2
HSA_SIGNAL_CONDITION_GTE = 3
hsa_signal_condition_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_wait_state_t'
hsa_wait_state_t__enumvalues = {
    0: 'HSA_WAIT_STATE_BLOCKED',
    1: 'HSA_WAIT_STATE_ACTIVE',
}
HSA_WAIT_STATE_BLOCKED = 0
HSA_WAIT_STATE_ACTIVE = 1
hsa_wait_state_t = ctypes.c_uint32 # enum
uint64_t = ctypes.c_uint64
try:
    hsa_signal_wait_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_wait_scacquire
    hsa_signal_wait_scacquire.restype = hsa_signal_value_t
    hsa_signal_wait_scacquire.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
try:
    hsa_signal_wait_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_wait_relaxed
    hsa_signal_wait_relaxed.restype = hsa_signal_value_t
    hsa_signal_wait_relaxed.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
try:
    hsa_signal_wait_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_wait_acquire
    hsa_signal_wait_acquire.restype = hsa_signal_value_t
    hsa_signal_wait_acquire.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
class struct_hsa_signal_group_s(Structure):
    pass

struct_hsa_signal_group_s._pack_ = 1 # source:False
struct_hsa_signal_group_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_signal_group_t = struct_hsa_signal_group_s
try:
    hsa_signal_group_create = _libraries['libhsa-runtime64.so'].hsa_signal_group_create
    hsa_signal_group_create.restype = hsa_status_t
    hsa_signal_group_create.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_signal_s), uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_signal_group_s)]
except AttributeError:
    pass
try:
    hsa_signal_group_destroy = _libraries['libhsa-runtime64.so'].hsa_signal_group_destroy
    hsa_signal_group_destroy.restype = hsa_status_t
    hsa_signal_group_destroy.argtypes = [hsa_signal_group_t]
except AttributeError:
    pass
try:
    hsa_signal_group_wait_any_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_group_wait_any_scacquire
    hsa_signal_group_wait_any_scacquire.restype = hsa_status_t
    hsa_signal_group_wait_any_scacquire.argtypes = [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), hsa_wait_state_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
try:
    hsa_signal_group_wait_any_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_group_wait_any_relaxed
    hsa_signal_group_wait_any_relaxed.restype = hsa_status_t
    hsa_signal_group_wait_any_relaxed.argtypes = [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), hsa_wait_state_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
class struct_hsa_region_s(Structure):
    pass

struct_hsa_region_s._pack_ = 1 # source:False
struct_hsa_region_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_region_t = struct_hsa_region_s

# values for enumeration 'hsa_queue_type_t'
hsa_queue_type_t__enumvalues = {
    0: 'HSA_QUEUE_TYPE_MULTI',
    1: 'HSA_QUEUE_TYPE_SINGLE',
    2: 'HSA_QUEUE_TYPE_COOPERATIVE',
}
HSA_QUEUE_TYPE_MULTI = 0
HSA_QUEUE_TYPE_SINGLE = 1
HSA_QUEUE_TYPE_COOPERATIVE = 2
hsa_queue_type_t = ctypes.c_uint32 # enum
hsa_queue_type32_t = ctypes.c_uint32

# values for enumeration 'hsa_queue_feature_t'
hsa_queue_feature_t__enumvalues = {
    1: 'HSA_QUEUE_FEATURE_KERNEL_DISPATCH',
    2: 'HSA_QUEUE_FEATURE_AGENT_DISPATCH',
}
HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1
HSA_QUEUE_FEATURE_AGENT_DISPATCH = 2
hsa_queue_feature_t = ctypes.c_uint32 # enum
class struct_hsa_queue_s(Structure):
    pass

struct_hsa_queue_s._pack_ = 1 # source:False
struct_hsa_queue_s._fields_ = [
    ('type', ctypes.c_uint32),
    ('features', ctypes.c_uint32),
    ('base_address', ctypes.POINTER(None)),
    ('doorbell_signal', hsa_signal_t),
    ('size', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('id', ctypes.c_uint64),
]

hsa_queue_t = struct_hsa_queue_s
try:
    hsa_queue_create = _libraries['libhsa-runtime64.so'].hsa_queue_create
    hsa_queue_create.restype = hsa_status_t
    hsa_queue_create.argtypes = [hsa_agent_t, uint32_t, hsa_queue_type32_t, ctypes.CFUNCTYPE(None, hsa_status_t, ctypes.POINTER(struct_hsa_queue_s), ctypes.POINTER(None)), ctypes.POINTER(None), uint32_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct_hsa_queue_s))]
except AttributeError:
    pass
try:
    hsa_soft_queue_create = _libraries['libhsa-runtime64.so'].hsa_soft_queue_create
    hsa_soft_queue_create.restype = hsa_status_t
    hsa_soft_queue_create.argtypes = [hsa_region_t, uint32_t, hsa_queue_type32_t, uint32_t, hsa_signal_t, ctypes.POINTER(ctypes.POINTER(struct_hsa_queue_s))]
except AttributeError:
    pass
try:
    hsa_queue_destroy = _libraries['libhsa-runtime64.so'].hsa_queue_destroy
    hsa_queue_destroy.restype = hsa_status_t
    hsa_queue_destroy.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_inactivate = _libraries['libhsa-runtime64.so'].hsa_queue_inactivate
    hsa_queue_inactivate.restype = hsa_status_t
    hsa_queue_inactivate.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_acquire
    hsa_queue_load_read_index_acquire.restype = uint64_t
    hsa_queue_load_read_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_scacquire
    hsa_queue_load_read_index_scacquire.restype = uint64_t
    hsa_queue_load_read_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_relaxed
    hsa_queue_load_read_index_relaxed.restype = uint64_t
    hsa_queue_load_read_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_acquire
    hsa_queue_load_write_index_acquire.restype = uint64_t
    hsa_queue_load_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_scacquire
    hsa_queue_load_write_index_scacquire.restype = uint64_t
    hsa_queue_load_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_relaxed
    hsa_queue_load_write_index_relaxed.restype = uint64_t
    hsa_queue_load_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_relaxed
    hsa_queue_store_write_index_relaxed.restype = None
    hsa_queue_store_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_release
    hsa_queue_store_write_index_release.restype = None
    hsa_queue_store_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_screlease
    hsa_queue_store_write_index_screlease.restype = None
    hsa_queue_store_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_acq_rel = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_acq_rel
    hsa_queue_cas_write_index_acq_rel.restype = uint64_t
    hsa_queue_cas_write_index_acq_rel.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_scacq_screl
    hsa_queue_cas_write_index_scacq_screl.restype = uint64_t
    hsa_queue_cas_write_index_scacq_screl.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_acquire
    hsa_queue_cas_write_index_acquire.restype = uint64_t
    hsa_queue_cas_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_scacquire
    hsa_queue_cas_write_index_scacquire.restype = uint64_t
    hsa_queue_cas_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_relaxed
    hsa_queue_cas_write_index_relaxed.restype = uint64_t
    hsa_queue_cas_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_release
    hsa_queue_cas_write_index_release.restype = uint64_t
    hsa_queue_cas_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_screlease
    hsa_queue_cas_write_index_screlease.restype = uint64_t
    hsa_queue_cas_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_acq_rel = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_acq_rel
    hsa_queue_add_write_index_acq_rel.restype = uint64_t
    hsa_queue_add_write_index_acq_rel.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_scacq_screl
    hsa_queue_add_write_index_scacq_screl.restype = uint64_t
    hsa_queue_add_write_index_scacq_screl.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_acquire
    hsa_queue_add_write_index_acquire.restype = uint64_t
    hsa_queue_add_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_scacquire
    hsa_queue_add_write_index_scacquire.restype = uint64_t
    hsa_queue_add_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_relaxed
    hsa_queue_add_write_index_relaxed.restype = uint64_t
    hsa_queue_add_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_release
    hsa_queue_add_write_index_release.restype = uint64_t
    hsa_queue_add_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_screlease
    hsa_queue_add_write_index_screlease.restype = uint64_t
    hsa_queue_add_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_relaxed
    hsa_queue_store_read_index_relaxed.restype = None
    hsa_queue_store_read_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_release
    hsa_queue_store_read_index_release.restype = None
    hsa_queue_store_read_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_screlease
    hsa_queue_store_read_index_screlease.restype = None
    hsa_queue_store_read_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass

# values for enumeration 'hsa_packet_type_t'
hsa_packet_type_t__enumvalues = {
    0: 'HSA_PACKET_TYPE_VENDOR_SPECIFIC',
    1: 'HSA_PACKET_TYPE_INVALID',
    2: 'HSA_PACKET_TYPE_KERNEL_DISPATCH',
    3: 'HSA_PACKET_TYPE_BARRIER_AND',
    4: 'HSA_PACKET_TYPE_AGENT_DISPATCH',
    5: 'HSA_PACKET_TYPE_BARRIER_OR',
}
HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0
HSA_PACKET_TYPE_INVALID = 1
HSA_PACKET_TYPE_KERNEL_DISPATCH = 2
HSA_PACKET_TYPE_BARRIER_AND = 3
HSA_PACKET_TYPE_AGENT_DISPATCH = 4
HSA_PACKET_TYPE_BARRIER_OR = 5
hsa_packet_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_fence_scope_t'
hsa_fence_scope_t__enumvalues = {
    0: 'HSA_FENCE_SCOPE_NONE',
    1: 'HSA_FENCE_SCOPE_AGENT',
    2: 'HSA_FENCE_SCOPE_SYSTEM',
}
HSA_FENCE_SCOPE_NONE = 0
HSA_FENCE_SCOPE_AGENT = 1
HSA_FENCE_SCOPE_SYSTEM = 2
hsa_fence_scope_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_packet_header_t'
hsa_packet_header_t__enumvalues = {
    0: 'HSA_PACKET_HEADER_TYPE',
    8: 'HSA_PACKET_HEADER_BARRIER',
    9: 'HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE',
    9: 'HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE',
    11: 'HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE',
    11: 'HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE',
}
HSA_PACKET_HEADER_TYPE = 0
HSA_PACKET_HEADER_BARRIER = 8
HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11
HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11
hsa_packet_header_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_packet_header_width_t'
hsa_packet_header_width_t__enumvalues = {
    8: 'HSA_PACKET_HEADER_WIDTH_TYPE',
    1: 'HSA_PACKET_HEADER_WIDTH_BARRIER',
    2: 'HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE',
}
HSA_PACKET_HEADER_WIDTH_TYPE = 8
HSA_PACKET_HEADER_WIDTH_BARRIER = 1
HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = 2
hsa_packet_header_width_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_kernel_dispatch_packet_setup_t'
hsa_kernel_dispatch_packet_setup_t__enumvalues = {
    0: 'HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS',
}
HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
hsa_kernel_dispatch_packet_setup_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_kernel_dispatch_packet_setup_width_t'
hsa_kernel_dispatch_packet_setup_width_t__enumvalues = {
    2: 'HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS',
}
HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2
hsa_kernel_dispatch_packet_setup_width_t = ctypes.c_uint32 # enum
class struct_hsa_kernel_dispatch_packet_s(Structure):
    pass

struct_hsa_kernel_dispatch_packet_s._pack_ = 1 # source:False
struct_hsa_kernel_dispatch_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('setup', ctypes.c_uint16),
    ('workgroup_size_x', ctypes.c_uint16),
    ('workgroup_size_y', ctypes.c_uint16),
    ('workgroup_size_z', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('grid_size_x', ctypes.c_uint32),
    ('grid_size_y', ctypes.c_uint32),
    ('grid_size_z', ctypes.c_uint32),
    ('private_segment_size', ctypes.c_uint32),
    ('group_segment_size', ctypes.c_uint32),
    ('kernel_object', ctypes.c_uint64),
    ('kernarg_address', ctypes.POINTER(None)),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_kernel_dispatch_packet_t = struct_hsa_kernel_dispatch_packet_s
class struct_hsa_agent_dispatch_packet_s(Structure):
    pass

struct_hsa_agent_dispatch_packet_s._pack_ = 1 # source:False
struct_hsa_agent_dispatch_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('type', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint32),
    ('return_address', ctypes.POINTER(None)),
    ('arg', ctypes.c_uint64 * 4),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_agent_dispatch_packet_t = struct_hsa_agent_dispatch_packet_s
class struct_hsa_barrier_and_packet_s(Structure):
    pass

struct_hsa_barrier_and_packet_s._pack_ = 1 # source:False
struct_hsa_barrier_and_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('reserved1', ctypes.c_uint32),
    ('dep_signal', struct_hsa_signal_s * 5),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_barrier_and_packet_t = struct_hsa_barrier_and_packet_s
class struct_hsa_barrier_or_packet_s(Structure):
    pass

struct_hsa_barrier_or_packet_s._pack_ = 1 # source:False
struct_hsa_barrier_or_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('reserved1', ctypes.c_uint32),
    ('dep_signal', struct_hsa_signal_s * 5),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_barrier_or_packet_t = struct_hsa_barrier_or_packet_s

# values for enumeration 'hsa_region_segment_t'
hsa_region_segment_t__enumvalues = {
    0: 'HSA_REGION_SEGMENT_GLOBAL',
    1: 'HSA_REGION_SEGMENT_READONLY',
    2: 'HSA_REGION_SEGMENT_PRIVATE',
    3: 'HSA_REGION_SEGMENT_GROUP',
    4: 'HSA_REGION_SEGMENT_KERNARG',
}
HSA_REGION_SEGMENT_GLOBAL = 0
HSA_REGION_SEGMENT_READONLY = 1
HSA_REGION_SEGMENT_PRIVATE = 2
HSA_REGION_SEGMENT_GROUP = 3
HSA_REGION_SEGMENT_KERNARG = 4
hsa_region_segment_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_region_global_flag_t'
hsa_region_global_flag_t__enumvalues = {
    1: 'HSA_REGION_GLOBAL_FLAG_KERNARG',
    2: 'HSA_REGION_GLOBAL_FLAG_FINE_GRAINED',
    4: 'HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED',
    8: 'HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
}
HSA_REGION_GLOBAL_FLAG_KERNARG = 1
HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = 2
HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = 4
HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8
hsa_region_global_flag_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_region_info_t'
hsa_region_info_t__enumvalues = {
    0: 'HSA_REGION_INFO_SEGMENT',
    1: 'HSA_REGION_INFO_GLOBAL_FLAGS',
    2: 'HSA_REGION_INFO_SIZE',
    4: 'HSA_REGION_INFO_ALLOC_MAX_SIZE',
    8: 'HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE',
    5: 'HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED',
    6: 'HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE',
    7: 'HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT',
}
HSA_REGION_INFO_SEGMENT = 0
HSA_REGION_INFO_GLOBAL_FLAGS = 1
HSA_REGION_INFO_SIZE = 2
HSA_REGION_INFO_ALLOC_MAX_SIZE = 4
HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = 8
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5
HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = 6
HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = 7
hsa_region_info_t = ctypes.c_uint32 # enum
try:
    hsa_region_get_info = _libraries['libhsa-runtime64.so'].hsa_region_get_info
    hsa_region_get_info.restype = hsa_status_t
    hsa_region_get_info.argtypes = [hsa_region_t, hsa_region_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_regions = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_regions
    hsa_agent_iterate_regions.restype = hsa_status_t
    hsa_agent_iterate_regions.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_region_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_memory_allocate = _libraries['libhsa-runtime64.so'].hsa_memory_allocate
    hsa_memory_allocate.restype = hsa_status_t
    hsa_memory_allocate.argtypes = [hsa_region_t, size_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_memory_free = _libraries['libhsa-runtime64.so'].hsa_memory_free
    hsa_memory_free.restype = hsa_status_t
    hsa_memory_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_memory_copy = _libraries['libhsa-runtime64.so'].hsa_memory_copy
    hsa_memory_copy.restype = hsa_status_t
    hsa_memory_copy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hsa_memory_assign_agent = _libraries['libhsa-runtime64.so'].hsa_memory_assign_agent
    hsa_memory_assign_agent.restype = hsa_status_t
    hsa_memory_assign_agent.argtypes = [ctypes.POINTER(None), hsa_agent_t, hsa_access_permission_t]
except AttributeError:
    pass
try:
    hsa_memory_register = _libraries['libhsa-runtime64.so'].hsa_memory_register
    hsa_memory_register.restype = hsa_status_t
    hsa_memory_register.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hsa_memory_deregister = _libraries['libhsa-runtime64.so'].hsa_memory_deregister
    hsa_memory_deregister.restype = hsa_status_t
    hsa_memory_deregister.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_isa_s(Structure):
    pass

struct_hsa_isa_s._pack_ = 1 # source:False
struct_hsa_isa_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_isa_t = struct_hsa_isa_s
try:
    hsa_isa_from_name = _libraries['libhsa-runtime64.so'].hsa_isa_from_name
    hsa_isa_from_name.restype = hsa_status_t
    hsa_isa_from_name.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_isa_s)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_isas = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_isas
    hsa_agent_iterate_isas.restype = hsa_status_t
    hsa_agent_iterate_isas.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_isa_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_isa_info_t'
hsa_isa_info_t__enumvalues = {
    0: 'HSA_ISA_INFO_NAME_LENGTH',
    1: 'HSA_ISA_INFO_NAME',
    2: 'HSA_ISA_INFO_CALL_CONVENTION_COUNT',
    3: 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE',
    4: 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT',
    5: 'HSA_ISA_INFO_MACHINE_MODELS',
    6: 'HSA_ISA_INFO_PROFILES',
    7: 'HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES',
    8: 'HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    9: 'HSA_ISA_INFO_FAST_F16_OPERATION',
    12: 'HSA_ISA_INFO_WORKGROUP_MAX_DIM',
    13: 'HSA_ISA_INFO_WORKGROUP_MAX_SIZE',
    14: 'HSA_ISA_INFO_GRID_MAX_DIM',
    16: 'HSA_ISA_INFO_GRID_MAX_SIZE',
    17: 'HSA_ISA_INFO_FBARRIER_MAX_SIZE',
}
HSA_ISA_INFO_NAME_LENGTH = 0
HSA_ISA_INFO_NAME = 1
HSA_ISA_INFO_CALL_CONVENTION_COUNT = 2
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = 3
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = 4
HSA_ISA_INFO_MACHINE_MODELS = 5
HSA_ISA_INFO_PROFILES = 6
HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7
HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8
HSA_ISA_INFO_FAST_F16_OPERATION = 9
HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12
HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13
HSA_ISA_INFO_GRID_MAX_DIM = 14
HSA_ISA_INFO_GRID_MAX_SIZE = 16
HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
hsa_isa_info_t = ctypes.c_uint32 # enum
try:
    hsa_isa_get_info = _libraries['libhsa-runtime64.so'].hsa_isa_get_info
    hsa_isa_get_info.restype = hsa_status_t
    hsa_isa_get_info.argtypes = [hsa_isa_t, hsa_isa_info_t, uint32_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_get_info_alt = _libraries['libhsa-runtime64.so'].hsa_isa_get_info_alt
    hsa_isa_get_info_alt.restype = hsa_status_t
    hsa_isa_get_info_alt.argtypes = [hsa_isa_t, hsa_isa_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_get_exception_policies = _libraries['libhsa-runtime64.so'].hsa_isa_get_exception_policies
    hsa_isa_get_exception_policies.restype = hsa_status_t
    hsa_isa_get_exception_policies.argtypes = [hsa_isa_t, hsa_profile_t, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass

# values for enumeration 'hsa_fp_type_t'
hsa_fp_type_t__enumvalues = {
    1: 'HSA_FP_TYPE_16',
    2: 'HSA_FP_TYPE_32',
    4: 'HSA_FP_TYPE_64',
}
HSA_FP_TYPE_16 = 1
HSA_FP_TYPE_32 = 2
HSA_FP_TYPE_64 = 4
hsa_fp_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_flush_mode_t'
hsa_flush_mode_t__enumvalues = {
    1: 'HSA_FLUSH_MODE_FTZ',
    2: 'HSA_FLUSH_MODE_NON_FTZ',
}
HSA_FLUSH_MODE_FTZ = 1
HSA_FLUSH_MODE_NON_FTZ = 2
hsa_flush_mode_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_round_method_t'
hsa_round_method_t__enumvalues = {
    1: 'HSA_ROUND_METHOD_SINGLE',
    2: 'HSA_ROUND_METHOD_DOUBLE',
}
HSA_ROUND_METHOD_SINGLE = 1
HSA_ROUND_METHOD_DOUBLE = 2
hsa_round_method_t = ctypes.c_uint32 # enum
try:
    hsa_isa_get_round_method = _libraries['libhsa-runtime64.so'].hsa_isa_get_round_method
    hsa_isa_get_round_method.restype = hsa_status_t
    hsa_isa_get_round_method.argtypes = [hsa_isa_t, hsa_fp_type_t, hsa_flush_mode_t, ctypes.POINTER(hsa_round_method_t)]
except AttributeError:
    pass
class struct_hsa_wavefront_s(Structure):
    pass

struct_hsa_wavefront_s._pack_ = 1 # source:False
struct_hsa_wavefront_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_wavefront_t = struct_hsa_wavefront_s

# values for enumeration 'hsa_wavefront_info_t'
hsa_wavefront_info_t__enumvalues = {
    0: 'HSA_WAVEFRONT_INFO_SIZE',
}
HSA_WAVEFRONT_INFO_SIZE = 0
hsa_wavefront_info_t = ctypes.c_uint32 # enum
try:
    hsa_wavefront_get_info = _libraries['libhsa-runtime64.so'].hsa_wavefront_get_info
    hsa_wavefront_get_info.restype = hsa_status_t
    hsa_wavefront_get_info.argtypes = [hsa_wavefront_t, hsa_wavefront_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_iterate_wavefronts = _libraries['libhsa-runtime64.so'].hsa_isa_iterate_wavefronts
    hsa_isa_iterate_wavefronts.restype = hsa_status_t
    hsa_isa_iterate_wavefronts.argtypes = [hsa_isa_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_wavefront_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_compatible = _libraries['libhsa-runtime64.so'].hsa_isa_compatible
    hsa_isa_compatible.restype = hsa_status_t
    hsa_isa_compatible.argtypes = [hsa_isa_t, hsa_isa_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
class struct_hsa_code_object_reader_s(Structure):
    pass

struct_hsa_code_object_reader_s._pack_ = 1 # source:False
struct_hsa_code_object_reader_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_object_reader_t = struct_hsa_code_object_reader_s
try:
    hsa_code_object_reader_create_from_file = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_create_from_file
    hsa_code_object_reader_create_from_file.restype = hsa_status_t
    hsa_code_object_reader_create_from_file.argtypes = [hsa_file_t, ctypes.POINTER(struct_hsa_code_object_reader_s)]
except AttributeError:
    pass
try:
    hsa_code_object_reader_create_from_memory = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_create_from_memory
    hsa_code_object_reader_create_from_memory.restype = hsa_status_t
    hsa_code_object_reader_create_from_memory.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_code_object_reader_s)]
except AttributeError:
    pass
try:
    hsa_code_object_reader_destroy = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_destroy
    hsa_code_object_reader_destroy.restype = hsa_status_t
    hsa_code_object_reader_destroy.argtypes = [hsa_code_object_reader_t]
except AttributeError:
    pass
class struct_hsa_executable_s(Structure):
    pass

struct_hsa_executable_s._pack_ = 1 # source:False
struct_hsa_executable_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_executable_t = struct_hsa_executable_s

# values for enumeration 'hsa_executable_state_t'
hsa_executable_state_t__enumvalues = {
    0: 'HSA_EXECUTABLE_STATE_UNFROZEN',
    1: 'HSA_EXECUTABLE_STATE_FROZEN',
}
HSA_EXECUTABLE_STATE_UNFROZEN = 0
HSA_EXECUTABLE_STATE_FROZEN = 1
hsa_executable_state_t = ctypes.c_uint32 # enum
try:
    hsa_executable_create = _libraries['libhsa-runtime64.so'].hsa_executable_create
    hsa_executable_create.restype = hsa_status_t
    hsa_executable_create.argtypes = [hsa_profile_t, hsa_executable_state_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_executable_s)]
except AttributeError:
    pass
try:
    hsa_executable_create_alt = _libraries['libhsa-runtime64.so'].hsa_executable_create_alt
    hsa_executable_create_alt.restype = hsa_status_t
    hsa_executable_create_alt.argtypes = [hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_executable_s)]
except AttributeError:
    pass
try:
    hsa_executable_destroy = _libraries['libhsa-runtime64.so'].hsa_executable_destroy
    hsa_executable_destroy.restype = hsa_status_t
    hsa_executable_destroy.argtypes = [hsa_executable_t]
except AttributeError:
    pass
class struct_hsa_loaded_code_object_s(Structure):
    pass

struct_hsa_loaded_code_object_s._pack_ = 1 # source:False
struct_hsa_loaded_code_object_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_loaded_code_object_t = struct_hsa_loaded_code_object_s
try:
    hsa_executable_load_program_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_program_code_object
    hsa_executable_load_program_code_object.restype = hsa_status_t
    hsa_executable_load_program_code_object.argtypes = [hsa_executable_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_loaded_code_object_s)]
except AttributeError:
    pass
try:
    hsa_executable_load_agent_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_agent_code_object
    hsa_executable_load_agent_code_object.restype = hsa_status_t
    hsa_executable_load_agent_code_object.argtypes = [hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_loaded_code_object_s)]
except AttributeError:
    pass
try:
    hsa_executable_freeze = _libraries['libhsa-runtime64.so'].hsa_executable_freeze
    hsa_executable_freeze.restype = hsa_status_t
    hsa_executable_freeze.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass

# values for enumeration 'hsa_executable_info_t'
hsa_executable_info_t__enumvalues = {
    1: 'HSA_EXECUTABLE_INFO_PROFILE',
    2: 'HSA_EXECUTABLE_INFO_STATE',
    3: 'HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_EXECUTABLE_INFO_PROFILE = 1
HSA_EXECUTABLE_INFO_STATE = 2
HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 3
hsa_executable_info_t = ctypes.c_uint32 # enum
try:
    hsa_executable_get_info = _libraries['libhsa-runtime64.so'].hsa_executable_get_info
    hsa_executable_get_info.restype = hsa_status_t
    hsa_executable_get_info.argtypes = [hsa_executable_t, hsa_executable_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_global_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_global_variable_define
    hsa_executable_global_variable_define.restype = hsa_status_t
    hsa_executable_global_variable_define.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_agent_global_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_agent_global_variable_define
    hsa_executable_agent_global_variable_define.restype = hsa_status_t
    hsa_executable_agent_global_variable_define.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_readonly_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_readonly_variable_define
    hsa_executable_readonly_variable_define.restype = hsa_status_t
    hsa_executable_readonly_variable_define.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_validate = _libraries['libhsa-runtime64.so'].hsa_executable_validate
    hsa_executable_validate.restype = hsa_status_t
    hsa_executable_validate.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_executable_validate_alt = _libraries['libhsa-runtime64.so'].hsa_executable_validate_alt
    hsa_executable_validate_alt.restype = hsa_status_t
    hsa_executable_validate_alt.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_executable_symbol_s(Structure):
    pass

struct_hsa_executable_symbol_s._pack_ = 1 # source:False
struct_hsa_executable_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_executable_symbol_t = struct_hsa_executable_symbol_s
int32_t = ctypes.c_int32
try:
    hsa_executable_get_symbol = _libraries['libhsa-runtime64.so'].hsa_executable_get_symbol
    hsa_executable_get_symbol.restype = hsa_status_t
    hsa_executable_get_symbol.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), hsa_agent_t, int32_t, ctypes.POINTER(struct_hsa_executable_symbol_s)]
except AttributeError:
    pass
try:
    hsa_executable_get_symbol_by_name = _libraries['libhsa-runtime64.so'].hsa_executable_get_symbol_by_name
    hsa_executable_get_symbol_by_name.restype = hsa_status_t
    hsa_executable_get_symbol_by_name.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_executable_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_symbol_kind_t'
hsa_symbol_kind_t__enumvalues = {
    0: 'HSA_SYMBOL_KIND_VARIABLE',
    1: 'HSA_SYMBOL_KIND_KERNEL',
    2: 'HSA_SYMBOL_KIND_INDIRECT_FUNCTION',
}
HSA_SYMBOL_KIND_VARIABLE = 0
HSA_SYMBOL_KIND_KERNEL = 1
HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2
hsa_symbol_kind_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_symbol_linkage_t'
hsa_symbol_linkage_t__enumvalues = {
    0: 'HSA_SYMBOL_LINKAGE_MODULE',
    1: 'HSA_SYMBOL_LINKAGE_PROGRAM',
}
HSA_SYMBOL_LINKAGE_MODULE = 0
HSA_SYMBOL_LINKAGE_PROGRAM = 1
hsa_symbol_linkage_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_variable_allocation_t'
hsa_variable_allocation_t__enumvalues = {
    0: 'HSA_VARIABLE_ALLOCATION_AGENT',
    1: 'HSA_VARIABLE_ALLOCATION_PROGRAM',
}
HSA_VARIABLE_ALLOCATION_AGENT = 0
HSA_VARIABLE_ALLOCATION_PROGRAM = 1
hsa_variable_allocation_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_variable_segment_t'
hsa_variable_segment_t__enumvalues = {
    0: 'HSA_VARIABLE_SEGMENT_GLOBAL',
    1: 'HSA_VARIABLE_SEGMENT_READONLY',
}
HSA_VARIABLE_SEGMENT_GLOBAL = 0
HSA_VARIABLE_SEGMENT_READONLY = 1
hsa_variable_segment_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_executable_symbol_info_t'
hsa_executable_symbol_info_t__enumvalues = {
    0: 'HSA_EXECUTABLE_SYMBOL_INFO_TYPE',
    1: 'HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH',
    2: 'HSA_EXECUTABLE_SYMBOL_INFO_NAME',
    3: 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    4: 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME',
    20: 'HSA_EXECUTABLE_SYMBOL_INFO_AGENT',
    21: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS',
    5: 'HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE',
    17: 'HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION',
    6: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    7: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT',
    8: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    9: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE',
    10: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST',
    22: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT',
    11: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    12: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    13: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    14: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    15: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    18: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    23: 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT',
    16: 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
}
HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0
HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1
HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = 4
HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21
HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5
HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = 7
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = 10
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = 23
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
hsa_executable_symbol_info_t = ctypes.c_uint32 # enum
try:
    hsa_executable_symbol_get_info = _libraries['libhsa-runtime64.so'].hsa_executable_symbol_get_info
    hsa_executable_symbol_get_info.restype = hsa_status_t
    hsa_executable_symbol_get_info.argtypes = [hsa_executable_symbol_t, hsa_executable_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_symbols
    hsa_executable_iterate_symbols.restype = hsa_status_t
    hsa_executable_iterate_symbols.argtypes = [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_agent_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_agent_symbols
    hsa_executable_iterate_agent_symbols.restype = hsa_status_t
    hsa_executable_iterate_agent_symbols.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_agent_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_program_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_program_symbols
    hsa_executable_iterate_program_symbols.restype = hsa_status_t
    hsa_executable_iterate_program_symbols.argtypes = [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_code_object_s(Structure):
    pass

struct_hsa_code_object_s._pack_ = 1 # source:False
struct_hsa_code_object_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_object_t = struct_hsa_code_object_s
class struct_hsa_callback_data_s(Structure):
    pass

struct_hsa_callback_data_s._pack_ = 1 # source:False
struct_hsa_callback_data_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_callback_data_t = struct_hsa_callback_data_s
try:
    hsa_code_object_serialize = _libraries['libhsa-runtime64.so'].hsa_code_object_serialize
    hsa_code_object_serialize.restype = hsa_status_t
    hsa_code_object_serialize.argtypes = [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, ctypes.c_uint64, struct_hsa_callback_data_s, ctypes.POINTER(ctypes.POINTER(None))), hsa_callback_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsa_code_object_deserialize = _libraries['libhsa-runtime64.so'].hsa_code_object_deserialize
    hsa_code_object_deserialize.restype = hsa_status_t
    hsa_code_object_deserialize.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_object_s)]
except AttributeError:
    pass
try:
    hsa_code_object_destroy = _libraries['libhsa-runtime64.so'].hsa_code_object_destroy
    hsa_code_object_destroy.restype = hsa_status_t
    hsa_code_object_destroy.argtypes = [hsa_code_object_t]
except AttributeError:
    pass

# values for enumeration 'hsa_code_object_type_t'
hsa_code_object_type_t__enumvalues = {
    0: 'HSA_CODE_OBJECT_TYPE_PROGRAM',
}
HSA_CODE_OBJECT_TYPE_PROGRAM = 0
hsa_code_object_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_code_object_info_t'
hsa_code_object_info_t__enumvalues = {
    0: 'HSA_CODE_OBJECT_INFO_VERSION',
    1: 'HSA_CODE_OBJECT_INFO_TYPE',
    2: 'HSA_CODE_OBJECT_INFO_ISA',
    3: 'HSA_CODE_OBJECT_INFO_MACHINE_MODEL',
    4: 'HSA_CODE_OBJECT_INFO_PROFILE',
    5: 'HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_CODE_OBJECT_INFO_VERSION = 0
HSA_CODE_OBJECT_INFO_TYPE = 1
HSA_CODE_OBJECT_INFO_ISA = 2
HSA_CODE_OBJECT_INFO_MACHINE_MODEL = 3
HSA_CODE_OBJECT_INFO_PROFILE = 4
HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
hsa_code_object_info_t = ctypes.c_uint32 # enum
try:
    hsa_code_object_get_info = _libraries['libhsa-runtime64.so'].hsa_code_object_get_info
    hsa_code_object_get_info.restype = hsa_status_t
    hsa_code_object_get_info.argtypes = [hsa_code_object_t, hsa_code_object_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_load_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_code_object
    hsa_executable_load_code_object.restype = hsa_status_t
    hsa_executable_load_code_object.argtypes = [hsa_executable_t, hsa_agent_t, hsa_code_object_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct_hsa_code_symbol_s(Structure):
    pass

struct_hsa_code_symbol_s._pack_ = 1 # source:False
struct_hsa_code_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_symbol_t = struct_hsa_code_symbol_s
try:
    hsa_code_object_get_symbol = _libraries['libhsa-runtime64.so'].hsa_code_object_get_symbol
    hsa_code_object_get_symbol.restype = hsa_status_t
    hsa_code_object_get_symbol.argtypes = [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_symbol_s)]
except AttributeError:
    pass
try:
    hsa_code_object_get_symbol_from_name = _libraries['libhsa-runtime64.so'].hsa_code_object_get_symbol_from_name
    hsa_code_object_get_symbol_from_name.restype = hsa_status_t
    hsa_code_object_get_symbol_from_name.argtypes = [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_code_symbol_info_t'
hsa_code_symbol_info_t__enumvalues = {
    0: 'HSA_CODE_SYMBOL_INFO_TYPE',
    1: 'HSA_CODE_SYMBOL_INFO_NAME_LENGTH',
    2: 'HSA_CODE_SYMBOL_INFO_NAME',
    3: 'HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    4: 'HSA_CODE_SYMBOL_INFO_MODULE_NAME',
    5: 'HSA_CODE_SYMBOL_INFO_LINKAGE',
    17: 'HSA_CODE_SYMBOL_INFO_IS_DEFINITION',
    6: 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    7: 'HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT',
    8: 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    9: 'HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE',
    10: 'HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST',
    11: 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    12: 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    13: 'HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    14: 'HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    15: 'HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    18: 'HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    16: 'HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
}
HSA_CODE_SYMBOL_INFO_TYPE = 0
HSA_CODE_SYMBOL_INFO_NAME_LENGTH = 1
HSA_CODE_SYMBOL_INFO_NAME = 2
HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3
HSA_CODE_SYMBOL_INFO_MODULE_NAME = 4
HSA_CODE_SYMBOL_INFO_LINKAGE = 5
HSA_CODE_SYMBOL_INFO_IS_DEFINITION = 17
HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6
HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT = 7
HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8
HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE = 9
HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST = 10
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12
HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13
HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14
HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15
HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18
HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
hsa_code_symbol_info_t = ctypes.c_uint32 # enum
try:
    hsa_code_symbol_get_info = _libraries['libhsa-runtime64.so'].hsa_code_symbol_get_info
    hsa_code_symbol_get_info.restype = hsa_status_t
    hsa_code_symbol_get_info.argtypes = [hsa_code_symbol_t, hsa_code_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_code_object_iterate_symbols = _libraries['libhsa-runtime64.so'].hsa_code_object_iterate_symbols
    hsa_code_object_iterate_symbols.restype = hsa_status_t
    hsa_code_object_iterate_symbols.argtypes = [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_code_object_s, struct_hsa_code_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'enum_hsa_ext_image_h_68'
enum_hsa_ext_image_h_68__enumvalues = {
    12288: 'HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED',
    12289: 'HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED',
    12290: 'HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED',
    12291: 'HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED',
}
HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = 12288
HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED = 12289
HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED = 12290
HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED = 12291
enum_hsa_ext_image_h_68 = ctypes.c_uint32 # enum

# values for enumeration 'enum_hsa_ext_image_h_93'
enum_hsa_ext_image_h_93__enumvalues = {
    12288: 'HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS',
    12289: 'HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS',
    12290: 'HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS',
    12291: 'HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS',
    12292: 'HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS',
    12293: 'HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS',
    12294: 'HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS',
    12295: 'HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS',
    12296: 'HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS',
    12297: 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES',
    12298: 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES',
    12299: 'HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS',
    12300: 'HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT',
}
HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS = 12288
HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS = 12289
HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS = 12290
HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS = 12291
HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS = 12292
HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS = 12293
HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS = 12294
HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS = 12295
HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS = 12296
HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES = 12297
HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES = 12298
HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS = 12299
HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT = 12300
enum_hsa_ext_image_h_93 = ctypes.c_uint32 # enum
class struct_hsa_ext_image_s(Structure):
    pass

struct_hsa_ext_image_s._pack_ = 1 # source:False
struct_hsa_ext_image_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_image_t = struct_hsa_ext_image_s

# values for enumeration 'hsa_ext_image_geometry_t'
hsa_ext_image_geometry_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_GEOMETRY_1D',
    1: 'HSA_EXT_IMAGE_GEOMETRY_2D',
    2: 'HSA_EXT_IMAGE_GEOMETRY_3D',
    3: 'HSA_EXT_IMAGE_GEOMETRY_1DA',
    4: 'HSA_EXT_IMAGE_GEOMETRY_2DA',
    5: 'HSA_EXT_IMAGE_GEOMETRY_1DB',
    6: 'HSA_EXT_IMAGE_GEOMETRY_2DDEPTH',
    7: 'HSA_EXT_IMAGE_GEOMETRY_2DADEPTH',
}
HSA_EXT_IMAGE_GEOMETRY_1D = 0
HSA_EXT_IMAGE_GEOMETRY_2D = 1
HSA_EXT_IMAGE_GEOMETRY_3D = 2
HSA_EXT_IMAGE_GEOMETRY_1DA = 3
HSA_EXT_IMAGE_GEOMETRY_2DA = 4
HSA_EXT_IMAGE_GEOMETRY_1DB = 5
HSA_EXT_IMAGE_GEOMETRY_2DDEPTH = 6
HSA_EXT_IMAGE_GEOMETRY_2DADEPTH = 7
hsa_ext_image_geometry_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_ext_image_channel_type_t'
hsa_ext_image_channel_type_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8',
    1: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16',
    2: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8',
    3: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16',
    4: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24',
    5: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555',
    6: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565',
    7: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010',
    8: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8',
    9: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16',
    10: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32',
    11: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8',
    12: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16',
    13: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32',
    14: 'HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT',
    15: 'HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT',
}
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 6
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = 7
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13
HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14
HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = 15
hsa_ext_image_channel_type_t = ctypes.c_uint32 # enum
hsa_ext_image_channel_type32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_image_channel_order_t'
hsa_ext_image_channel_order_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CHANNEL_ORDER_A',
    1: 'HSA_EXT_IMAGE_CHANNEL_ORDER_R',
    2: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RX',
    3: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RG',
    4: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGX',
    5: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RA',
    6: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGB',
    7: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX',
    8: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA',
    9: 'HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA',
    10: 'HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB',
    11: 'HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR',
    12: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB',
    13: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX',
    14: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA',
    15: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA',
    16: 'HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY',
    17: 'HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE',
    18: 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH',
    19: 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL',
}
HSA_EXT_IMAGE_CHANNEL_ORDER_A = 0
HSA_EXT_IMAGE_CHANNEL_ORDER_R = 1
HSA_EXT_IMAGE_CHANNEL_ORDER_RX = 2
HSA_EXT_IMAGE_CHANNEL_ORDER_RG = 3
HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = 4
HSA_EXT_IMAGE_CHANNEL_ORDER_RA = 5
HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = 6
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = 7
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = 8
HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = 9
HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = 10
HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = 11
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = 12
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = 13
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = 14
HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = 15
HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = 16
HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = 17
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = 18
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
hsa_ext_image_channel_order_t = ctypes.c_uint32 # enum
hsa_ext_image_channel_order32_t = ctypes.c_uint32
class struct_hsa_ext_image_format_s(Structure):
    pass

struct_hsa_ext_image_format_s._pack_ = 1 # source:False
struct_hsa_ext_image_format_s._fields_ = [
    ('channel_type', ctypes.c_uint32),
    ('channel_order', ctypes.c_uint32),
]

hsa_ext_image_format_t = struct_hsa_ext_image_format_s
class struct_hsa_ext_image_descriptor_s(Structure):
    pass

struct_hsa_ext_image_descriptor_s._pack_ = 1 # source:False
struct_hsa_ext_image_descriptor_s._fields_ = [
    ('geometry', hsa_ext_image_geometry_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('array_size', ctypes.c_uint64),
    ('format', hsa_ext_image_format_t),
]

hsa_ext_image_descriptor_t = struct_hsa_ext_image_descriptor_s

# values for enumeration 'hsa_ext_image_capability_t'
hsa_ext_image_capability_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED',
    1: 'HSA_EXT_IMAGE_CAPABILITY_READ_ONLY',
    2: 'HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY',
    4: 'HSA_EXT_IMAGE_CAPABILITY_READ_WRITE',
    8: 'HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE',
    16: 'HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT',
}
HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED = 0
HSA_EXT_IMAGE_CAPABILITY_READ_ONLY = 1
HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY = 2
HSA_EXT_IMAGE_CAPABILITY_READ_WRITE = 4
HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE = 8
HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT = 16
hsa_ext_image_capability_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_ext_image_data_layout_t'
hsa_ext_image_data_layout_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE',
    1: 'HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR',
}
HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE = 0
HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR = 1
hsa_ext_image_data_layout_t = ctypes.c_uint32 # enum
try:
    hsa_ext_image_get_capability = _libraries['libhsa-runtime64.so'].hsa_ext_image_get_capability
    hsa_ext_image_get_capability.restype = hsa_status_t
    hsa_ext_image_get_capability.argtypes = [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_ext_image_get_capability_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_get_capability_with_layout
    hsa_ext_image_get_capability_with_layout.restype = hsa_status_t
    hsa_ext_image_get_capability_with_layout.argtypes = [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), hsa_ext_image_data_layout_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_ext_image_data_info_s(Structure):
    pass

struct_hsa_ext_image_data_info_s._pack_ = 1 # source:False
struct_hsa_ext_image_data_info_s._fields_ = [
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
]

hsa_ext_image_data_info_t = struct_hsa_ext_image_data_info_s
try:
    hsa_ext_image_data_get_info = _libraries['libhsa-runtime64.so'].hsa_ext_image_data_get_info
    hsa_ext_image_data_get_info.restype = hsa_status_t
    hsa_ext_image_data_get_info.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_data_get_info_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_data_get_info_with_layout
    hsa_ext_image_data_get_info_with_layout.restype = hsa_status_t
    hsa_ext_image_data_get_info_with_layout.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_create = _libraries['libhsa-runtime64.so'].hsa_ext_image_create
    hsa_ext_image_create.restype = hsa_status_t
    hsa_ext_image_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_create_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_create_with_layout
    hsa_ext_image_create_with_layout.restype = hsa_status_t
    hsa_ext_image_create_with_layout.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_image_destroy
    hsa_ext_image_destroy.restype = hsa_status_t
    hsa_ext_image_destroy.argtypes = [hsa_agent_t, hsa_ext_image_t]
except AttributeError:
    pass
try:
    hsa_ext_image_copy = _libraries['libhsa-runtime64.so'].hsa_ext_image_copy
    hsa_ext_image_copy.restype = hsa_status_t
    hsa_ext_image_copy.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(struct_hsa_dim3_s), hsa_ext_image_t, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s)]
except AttributeError:
    pass
class struct_hsa_ext_image_region_s(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('offset', hsa_dim3_t),
    ('range', hsa_dim3_t),
     ]

hsa_ext_image_region_t = struct_hsa_ext_image_region_s
try:
    hsa_ext_image_import = _libraries['libhsa-runtime64.so'].hsa_ext_image_import
    hsa_ext_image_import.restype = hsa_status_t
    hsa_ext_image_import.argtypes = [hsa_agent_t, ctypes.POINTER(None), size_t, size_t, hsa_ext_image_t, ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_export = _libraries['libhsa-runtime64.so'].hsa_ext_image_export
    hsa_ext_image_export.restype = hsa_status_t
    hsa_ext_image_export.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(None), size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_clear = _libraries['libhsa-runtime64.so'].hsa_ext_image_clear
    hsa_ext_image_clear.restype = hsa_status_t
    hsa_ext_image_clear.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
class struct_hsa_ext_sampler_s(Structure):
    pass

struct_hsa_ext_sampler_s._pack_ = 1 # source:False
struct_hsa_ext_sampler_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_sampler_t = struct_hsa_ext_sampler_s

# values for enumeration 'hsa_ext_sampler_addressing_mode_t'
hsa_ext_sampler_addressing_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED',
    1: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE',
    2: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER',
    3: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT',
    4: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT',
}
HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED = 0
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = 1
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER = 2
HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT = 3
HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = 4
hsa_ext_sampler_addressing_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_addressing_mode32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_sampler_coordinate_mode_t'
hsa_ext_sampler_coordinate_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED',
    1: 'HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED',
}
HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED = 0
HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED = 1
hsa_ext_sampler_coordinate_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_coordinate_mode32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_sampler_filter_mode_t'
hsa_ext_sampler_filter_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_FILTER_MODE_NEAREST',
    1: 'HSA_EXT_SAMPLER_FILTER_MODE_LINEAR',
}
HSA_EXT_SAMPLER_FILTER_MODE_NEAREST = 0
HSA_EXT_SAMPLER_FILTER_MODE_LINEAR = 1
hsa_ext_sampler_filter_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_filter_mode32_t = ctypes.c_uint32
class struct_hsa_ext_sampler_descriptor_s(Structure):
    pass

struct_hsa_ext_sampler_descriptor_s._pack_ = 1 # source:False
struct_hsa_ext_sampler_descriptor_s._fields_ = [
    ('coordinate_mode', ctypes.c_uint32),
    ('filter_mode', ctypes.c_uint32),
    ('address_mode', ctypes.c_uint32),
]

hsa_ext_sampler_descriptor_t = struct_hsa_ext_sampler_descriptor_s
try:
    hsa_ext_sampler_create = _libraries['libhsa-runtime64.so'].hsa_ext_sampler_create
    hsa_ext_sampler_create.restype = hsa_status_t
    hsa_ext_sampler_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s)]
except AttributeError:
    pass
try:
    hsa_ext_sampler_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_sampler_destroy
    hsa_ext_sampler_destroy.restype = hsa_status_t
    hsa_ext_sampler_destroy.argtypes = [hsa_agent_t, hsa_ext_sampler_t]
except AttributeError:
    pass
class struct_hsa_ext_images_1_00_pfn_s(Structure):
    pass

struct_hsa_ext_images_1_00_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_images_1_00_pfn_s._fields_ = [
    ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s))),
    ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s)),
    ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s))),
    ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s))),
    ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_sampler_s)),
]

hsa_ext_images_1_00_pfn_t = struct_hsa_ext_images_1_00_pfn_s
class struct_hsa_ext_images_1_pfn_s(Structure):
    pass

struct_hsa_ext_images_1_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_images_1_pfn_s._fields_ = [
    ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s))),
    ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s)),
    ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s))),
    ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s))),
    ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_sampler_s)),
    ('hsa_ext_image_get_capability_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), hsa_ext_image_data_layout_t, ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, hsa_ext_image_data_layout_t, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, hsa_ext_image_data_layout_t, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_s))),
]

hsa_ext_images_1_pfn_t = struct_hsa_ext_images_1_pfn_s
hsa_signal_condition32_t = ctypes.c_uint32

# values for enumeration 'hsa_amd_packet_type_t'
hsa_amd_packet_type_t__enumvalues = {
    2: 'HSA_AMD_PACKET_TYPE_BARRIER_VALUE',
}
HSA_AMD_PACKET_TYPE_BARRIER_VALUE = 2
hsa_amd_packet_type_t = ctypes.c_uint32 # enum
hsa_amd_packet_type8_t = ctypes.c_ubyte
class struct_hsa_amd_packet_header_s(Structure):
    pass

struct_hsa_amd_packet_header_s._pack_ = 1 # source:False
struct_hsa_amd_packet_header_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('AmdFormat', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

hsa_amd_vendor_packet_header_t = struct_hsa_amd_packet_header_s
class struct_hsa_amd_barrier_value_packet_s(Structure):
    pass

struct_hsa_amd_barrier_value_packet_s._pack_ = 1 # source:False
struct_hsa_amd_barrier_value_packet_s._fields_ = [
    ('header', hsa_amd_vendor_packet_header_t),
    ('reserved0', ctypes.c_uint32),
    ('signal', hsa_signal_t),
    ('value', ctypes.c_int64),
    ('mask', ctypes.c_int64),
    ('cond', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint64),
    ('reserved3', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_amd_barrier_value_packet_t = struct_hsa_amd_barrier_value_packet_s

# values for enumeration 'enum_hsa_ext_amd_h_179'
enum_hsa_ext_amd_h_179__enumvalues = {
    40: 'HSA_STATUS_ERROR_INVALID_MEMORY_POOL',
    41: 'HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION',
    42: 'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION',
    43: 'HSA_STATUS_ERROR_MEMORY_FAULT',
    44: 'HSA_STATUS_CU_MASK_REDUCED',
    45: 'HSA_STATUS_ERROR_OUT_OF_REGISTERS',
}
HSA_STATUS_ERROR_INVALID_MEMORY_POOL = 40
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = 41
HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = 42
HSA_STATUS_ERROR_MEMORY_FAULT = 43
HSA_STATUS_CU_MASK_REDUCED = 44
HSA_STATUS_ERROR_OUT_OF_REGISTERS = 45
enum_hsa_ext_amd_h_179 = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_iommu_version_t'
hsa_amd_iommu_version_t__enumvalues = {
    0: 'HSA_IOMMU_SUPPORT_NONE',
    1: 'HSA_IOMMU_SUPPORT_V2',
}
HSA_IOMMU_SUPPORT_NONE = 0
HSA_IOMMU_SUPPORT_V2 = 1
hsa_amd_iommu_version_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_agent_info_s'
hsa_amd_agent_info_s__enumvalues = {
    40960: 'HSA_AMD_AGENT_INFO_CHIP_ID',
    40961: 'HSA_AMD_AGENT_INFO_CACHELINE_SIZE',
    40962: 'HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT',
    40963: 'HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY',
    40964: 'HSA_AMD_AGENT_INFO_DRIVER_NODE_ID',
    40965: 'HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS',
    40966: 'HSA_AMD_AGENT_INFO_BDFID',
    40967: 'HSA_AMD_AGENT_INFO_MEMORY_WIDTH',
    40968: 'HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY',
    40969: 'HSA_AMD_AGENT_INFO_PRODUCT_NAME',
    40970: 'HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU',
    40971: 'HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU',
    40972: 'HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES',
    40973: 'HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE',
    40974: 'HSA_AMD_AGENT_INFO_HDP_FLUSH',
    40975: 'HSA_AMD_AGENT_INFO_DOMAIN',
    40976: 'HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES',
    40977: 'HSA_AMD_AGENT_INFO_UUID',
    40978: 'HSA_AMD_AGENT_INFO_ASIC_REVISION',
    40979: 'HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS',
    40980: 'HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT',
    40981: 'HSA_AMD_AGENT_INFO_MEMORY_AVAIL',
    40982: 'HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY',
    41223: 'HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID',
    41224: 'HSA_AMD_AGENT_INFO_UCODE_VERSION',
    41225: 'HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION',
    41226: 'HSA_AMD_AGENT_INFO_NUM_SDMA_ENG',
    41227: 'HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG',
    41232: 'HSA_AMD_AGENT_INFO_IOMMU_SUPPORT',
    41233: 'HSA_AMD_AGENT_INFO_NUM_XCC',
    41234: 'HSA_AMD_AGENT_INFO_DRIVER_UID',
    41235: 'HSA_AMD_AGENT_INFO_NEAREST_CPU',
}
HSA_AMD_AGENT_INFO_CHIP_ID = 40960
HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 40961
HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 40962
HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 40963
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 40964
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = 40965
HSA_AMD_AGENT_INFO_BDFID = 40966
HSA_AMD_AGENT_INFO_MEMORY_WIDTH = 40967
HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = 40968
HSA_AMD_AGENT_INFO_PRODUCT_NAME = 40969
HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 40970
HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = 40971
HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = 40972
HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = 40973
HSA_AMD_AGENT_INFO_HDP_FLUSH = 40974
HSA_AMD_AGENT_INFO_DOMAIN = 40975
HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = 40976
HSA_AMD_AGENT_INFO_UUID = 40977
HSA_AMD_AGENT_INFO_ASIC_REVISION = 40978
HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = 40979
HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = 40980
HSA_AMD_AGENT_INFO_MEMORY_AVAIL = 40981
HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 40982
HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID = 41223
HSA_AMD_AGENT_INFO_UCODE_VERSION = 41224
HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION = 41225
HSA_AMD_AGENT_INFO_NUM_SDMA_ENG = 41226
HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG = 41227
HSA_AMD_AGENT_INFO_IOMMU_SUPPORT = 41232
HSA_AMD_AGENT_INFO_NUM_XCC = 41233
HSA_AMD_AGENT_INFO_DRIVER_UID = 41234
HSA_AMD_AGENT_INFO_NEAREST_CPU = 41235
hsa_amd_agent_info_s = ctypes.c_uint32 # enum
hsa_amd_agent_info_t = hsa_amd_agent_info_s
hsa_amd_agent_info_t__enumvalues = hsa_amd_agent_info_s__enumvalues

# values for enumeration 'hsa_amd_sdma_engine_id'
hsa_amd_sdma_engine_id__enumvalues = {
    1: 'HSA_AMD_SDMA_ENGINE_0',
    2: 'HSA_AMD_SDMA_ENGINE_1',
    4: 'HSA_AMD_SDMA_ENGINE_2',
    8: 'HSA_AMD_SDMA_ENGINE_3',
    16: 'HSA_AMD_SDMA_ENGINE_4',
    32: 'HSA_AMD_SDMA_ENGINE_5',
    64: 'HSA_AMD_SDMA_ENGINE_6',
    128: 'HSA_AMD_SDMA_ENGINE_7',
    256: 'HSA_AMD_SDMA_ENGINE_8',
    512: 'HSA_AMD_SDMA_ENGINE_9',
    1024: 'HSA_AMD_SDMA_ENGINE_10',
    2048: 'HSA_AMD_SDMA_ENGINE_11',
    4096: 'HSA_AMD_SDMA_ENGINE_12',
    8192: 'HSA_AMD_SDMA_ENGINE_13',
    16384: 'HSA_AMD_SDMA_ENGINE_14',
    32768: 'HSA_AMD_SDMA_ENGINE_15',
}
HSA_AMD_SDMA_ENGINE_0 = 1
HSA_AMD_SDMA_ENGINE_1 = 2
HSA_AMD_SDMA_ENGINE_2 = 4
HSA_AMD_SDMA_ENGINE_3 = 8
HSA_AMD_SDMA_ENGINE_4 = 16
HSA_AMD_SDMA_ENGINE_5 = 32
HSA_AMD_SDMA_ENGINE_6 = 64
HSA_AMD_SDMA_ENGINE_7 = 128
HSA_AMD_SDMA_ENGINE_8 = 256
HSA_AMD_SDMA_ENGINE_9 = 512
HSA_AMD_SDMA_ENGINE_10 = 1024
HSA_AMD_SDMA_ENGINE_11 = 2048
HSA_AMD_SDMA_ENGINE_12 = 4096
HSA_AMD_SDMA_ENGINE_13 = 8192
HSA_AMD_SDMA_ENGINE_14 = 16384
HSA_AMD_SDMA_ENGINE_15 = 32768
hsa_amd_sdma_engine_id = ctypes.c_uint32 # enum
hsa_amd_sdma_engine_id_t = hsa_amd_sdma_engine_id
hsa_amd_sdma_engine_id_t__enumvalues = hsa_amd_sdma_engine_id__enumvalues
class struct_hsa_amd_hdp_flush_s(Structure):
    pass

struct_hsa_amd_hdp_flush_s._pack_ = 1 # source:False
struct_hsa_amd_hdp_flush_s._fields_ = [
    ('HDP_MEM_FLUSH_CNTL', ctypes.POINTER(ctypes.c_uint32)),
    ('HDP_REG_FLUSH_CNTL', ctypes.POINTER(ctypes.c_uint32)),
]

hsa_amd_hdp_flush_t = struct_hsa_amd_hdp_flush_s

# values for enumeration 'hsa_amd_region_info_s'
hsa_amd_region_info_s__enumvalues = {
    40960: 'HSA_AMD_REGION_INFO_HOST_ACCESSIBLE',
    40961: 'HSA_AMD_REGION_INFO_BASE',
    40962: 'HSA_AMD_REGION_INFO_BUS_WIDTH',
    40963: 'HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY',
}
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = 40960
HSA_AMD_REGION_INFO_BASE = 40961
HSA_AMD_REGION_INFO_BUS_WIDTH = 40962
HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = 40963
hsa_amd_region_info_s = ctypes.c_uint32 # enum
hsa_amd_region_info_t = hsa_amd_region_info_s
hsa_amd_region_info_t__enumvalues = hsa_amd_region_info_s__enumvalues

# values for enumeration 'hsa_amd_coherency_type_s'
hsa_amd_coherency_type_s__enumvalues = {
    0: 'HSA_AMD_COHERENCY_TYPE_COHERENT',
    1: 'HSA_AMD_COHERENCY_TYPE_NONCOHERENT',
}
HSA_AMD_COHERENCY_TYPE_COHERENT = 0
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = 1
hsa_amd_coherency_type_s = ctypes.c_uint32 # enum
hsa_amd_coherency_type_t = hsa_amd_coherency_type_s
hsa_amd_coherency_type_t__enumvalues = hsa_amd_coherency_type_s__enumvalues
try:
    hsa_amd_coherency_get_type = _libraries['libhsa-runtime64.so'].hsa_amd_coherency_get_type
    hsa_amd_coherency_get_type.restype = hsa_status_t
    hsa_amd_coherency_get_type.argtypes = [hsa_agent_t, ctypes.POINTER(hsa_amd_coherency_type_s)]
except AttributeError:
    pass
try:
    hsa_amd_coherency_set_type = _libraries['libhsa-runtime64.so'].hsa_amd_coherency_set_type
    hsa_amd_coherency_set_type.restype = hsa_status_t
    hsa_amd_coherency_set_type.argtypes = [hsa_agent_t, hsa_amd_coherency_type_t]
except AttributeError:
    pass
class struct_hsa_amd_profiling_dispatch_time_s(Structure):
    pass

struct_hsa_amd_profiling_dispatch_time_s._pack_ = 1 # source:False
struct_hsa_amd_profiling_dispatch_time_s._fields_ = [
    ('start', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

hsa_amd_profiling_dispatch_time_t = struct_hsa_amd_profiling_dispatch_time_s
class struct_hsa_amd_profiling_async_copy_time_s(Structure):
    pass

struct_hsa_amd_profiling_async_copy_time_s._pack_ = 1 # source:False
struct_hsa_amd_profiling_async_copy_time_s._fields_ = [
    ('start', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

hsa_amd_profiling_async_copy_time_t = struct_hsa_amd_profiling_async_copy_time_s
try:
    hsa_amd_profiling_set_profiler_enabled = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_set_profiler_enabled
    hsa_amd_profiling_set_profiler_enabled.restype = hsa_status_t
    hsa_amd_profiling_set_profiler_enabled.argtypes = [ctypes.POINTER(struct_hsa_queue_s), ctypes.c_int32]
except AttributeError:
    pass
try:
    hsa_amd_profiling_async_copy_enable = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_async_copy_enable
    hsa_amd_profiling_async_copy_enable.restype = hsa_status_t
    hsa_amd_profiling_async_copy_enable.argtypes = [ctypes.c_bool]
except AttributeError:
    pass
try:
    hsa_amd_profiling_get_dispatch_time = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_get_dispatch_time
    hsa_amd_profiling_get_dispatch_time.restype = hsa_status_t
    hsa_amd_profiling_get_dispatch_time.argtypes = [hsa_agent_t, hsa_signal_t, ctypes.POINTER(struct_hsa_amd_profiling_dispatch_time_s)]
except AttributeError:
    pass
try:
    hsa_amd_profiling_get_async_copy_time = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_get_async_copy_time
    hsa_amd_profiling_get_async_copy_time.restype = hsa_status_t
    hsa_amd_profiling_get_async_copy_time.argtypes = [hsa_signal_t, ctypes.POINTER(struct_hsa_amd_profiling_async_copy_time_s)]
except AttributeError:
    pass
try:
    hsa_amd_profiling_convert_tick_to_system_domain = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_convert_tick_to_system_domain
    hsa_amd_profiling_convert_tick_to_system_domain.restype = hsa_status_t
    hsa_amd_profiling_convert_tick_to_system_domain.argtypes = [hsa_agent_t, uint64_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_signal_attribute_t'
hsa_amd_signal_attribute_t__enumvalues = {
    1: 'HSA_AMD_SIGNAL_AMD_GPU_ONLY',
    2: 'HSA_AMD_SIGNAL_IPC',
}
HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1
HSA_AMD_SIGNAL_IPC = 2
hsa_amd_signal_attribute_t = ctypes.c_uint32 # enum
try:
    hsa_amd_signal_create = _libraries['libhsa-runtime64.so'].hsa_amd_signal_create
    hsa_amd_signal_create.restype = hsa_status_t
    hsa_amd_signal_create.argtypes = [hsa_signal_value_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), uint64_t, ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass
try:
    hsa_amd_signal_value_pointer = _libraries['libhsa-runtime64.so'].hsa_amd_signal_value_pointer
    hsa_amd_signal_value_pointer.restype = hsa_status_t
    hsa_amd_signal_value_pointer.argtypes = [hsa_signal_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))]
except AttributeError:
    pass
hsa_amd_signal_handler = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int64, ctypes.POINTER(None))
try:
    hsa_amd_signal_async_handler = _libraries['libhsa-runtime64.so'].hsa_amd_signal_async_handler
    hsa_amd_signal_async_handler.restype = hsa_status_t
    hsa_amd_signal_async_handler.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, hsa_amd_signal_handler, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_async_function = _libraries['libhsa-runtime64.so'].hsa_amd_async_function
    hsa_amd_async_function.restype = hsa_status_t
    hsa_amd_async_function.argtypes = [ctypes.CFUNCTYPE(None, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_signal_wait_any = _libraries['libhsa-runtime64.so'].hsa_amd_signal_wait_any
    hsa_amd_signal_wait_any.restype = uint32_t
    hsa_amd_signal_wait_any.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), uint64_t, hsa_wait_state_t, ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
try:
    hsa_amd_image_get_info_max_dim = _libraries['libhsa-runtime64.so'].hsa_amd_image_get_info_max_dim
    hsa_amd_image_get_info_max_dim.restype = hsa_status_t
    hsa_amd_image_get_info_max_dim.argtypes = [hsa_agent_t, hsa_agent_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_queue_cu_set_mask = _libraries['libhsa-runtime64.so'].hsa_amd_queue_cu_set_mask
    hsa_amd_queue_cu_set_mask.restype = hsa_status_t
    hsa_amd_queue_cu_set_mask.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_amd_queue_cu_get_mask = _libraries['libhsa-runtime64.so'].hsa_amd_queue_cu_get_mask
    hsa_amd_queue_cu_get_mask.restype = hsa_status_t
    hsa_amd_queue_cu_get_mask.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_segment_t'
hsa_amd_segment_t__enumvalues = {
    0: 'HSA_AMD_SEGMENT_GLOBAL',
    1: 'HSA_AMD_SEGMENT_READONLY',
    2: 'HSA_AMD_SEGMENT_PRIVATE',
    3: 'HSA_AMD_SEGMENT_GROUP',
}
HSA_AMD_SEGMENT_GLOBAL = 0
HSA_AMD_SEGMENT_READONLY = 1
HSA_AMD_SEGMENT_PRIVATE = 2
HSA_AMD_SEGMENT_GROUP = 3
hsa_amd_segment_t = ctypes.c_uint32 # enum
class struct_hsa_amd_memory_pool_s(Structure):
    pass

struct_hsa_amd_memory_pool_s._pack_ = 1 # source:False
struct_hsa_amd_memory_pool_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_amd_memory_pool_t = struct_hsa_amd_memory_pool_s

# values for enumeration 'hsa_amd_memory_pool_global_flag_s'
hsa_amd_memory_pool_global_flag_s__enumvalues = {
    1: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT',
    2: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED',
    4: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED',
    8: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
}
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8
hsa_amd_memory_pool_global_flag_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_global_flag_t = hsa_amd_memory_pool_global_flag_s
hsa_amd_memory_pool_global_flag_t__enumvalues = hsa_amd_memory_pool_global_flag_s__enumvalues

# values for enumeration 'hsa_amd_memory_pool_location_s'
hsa_amd_memory_pool_location_s__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_LOCATION_CPU',
    1: 'HSA_AMD_MEMORY_POOL_LOCATION_GPU',
}
HSA_AMD_MEMORY_POOL_LOCATION_CPU = 0
HSA_AMD_MEMORY_POOL_LOCATION_GPU = 1
hsa_amd_memory_pool_location_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_location_t = hsa_amd_memory_pool_location_s
hsa_amd_memory_pool_location_t__enumvalues = hsa_amd_memory_pool_location_s__enumvalues

# values for enumeration 'hsa_amd_memory_pool_info_t'
hsa_amd_memory_pool_info_t__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_INFO_SEGMENT',
    1: 'HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS',
    2: 'HSA_AMD_MEMORY_POOL_INFO_SIZE',
    5: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED',
    6: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE',
    7: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT',
    15: 'HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL',
    16: 'HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE',
    17: 'HSA_AMD_MEMORY_POOL_INFO_LOCATION',
    18: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE',
}
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1
HSA_AMD_MEMORY_POOL_INFO_SIZE = 2
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7
HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15
HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = 16
HSA_AMD_MEMORY_POOL_INFO_LOCATION = 17
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18
hsa_amd_memory_pool_info_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_memory_pool_flag_s'
hsa_amd_memory_pool_flag_s__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_STANDARD_FLAG',
    1: 'HSA_AMD_MEMORY_POOL_PCIE_FLAG',
}
HSA_AMD_MEMORY_POOL_STANDARD_FLAG = 0
HSA_AMD_MEMORY_POOL_PCIE_FLAG = 1
hsa_amd_memory_pool_flag_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_flag_t = hsa_amd_memory_pool_flag_s
hsa_amd_memory_pool_flag_t__enumvalues = hsa_amd_memory_pool_flag_s__enumvalues
try:
    hsa_amd_memory_pool_get_info = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_get_info
    hsa_amd_memory_pool_get_info.restype = hsa_status_t
    hsa_amd_memory_pool_get_info.argtypes = [hsa_amd_memory_pool_t, hsa_amd_memory_pool_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_agent_iterate_memory_pools = _libraries['libhsa-runtime64.so'].hsa_amd_agent_iterate_memory_pools
    hsa_amd_agent_iterate_memory_pools.restype = hsa_status_t
    hsa_amd_agent_iterate_memory_pools.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_amd_memory_pool_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_allocate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_allocate
    hsa_amd_memory_pool_allocate.restype = hsa_status_t
    hsa_amd_memory_pool_allocate.argtypes = [hsa_amd_memory_pool_t, size_t, uint32_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_free = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_free
    hsa_amd_memory_pool_free.restype = hsa_status_t
    hsa_amd_memory_pool_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_async_copy = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy
    hsa_amd_memory_async_copy.restype = hsa_status_t
    hsa_amd_memory_async_copy.argtypes = [ctypes.POINTER(None), hsa_agent_t, ctypes.POINTER(None), hsa_agent_t, size_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_amd_memory_async_copy_on_engine = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy_on_engine
    hsa_amd_memory_async_copy_on_engine.restype = hsa_status_t
    hsa_amd_memory_async_copy_on_engine.argtypes = [ctypes.POINTER(None), hsa_agent_t, ctypes.POINTER(None), hsa_agent_t, size_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t, hsa_amd_sdma_engine_id_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    hsa_amd_memory_copy_engine_status = _libraries['libhsa-runtime64.so'].hsa_amd_memory_copy_engine_status
    hsa_amd_memory_copy_engine_status.restype = hsa_status_t
    hsa_amd_memory_copy_engine_status.argtypes = [hsa_agent_t, hsa_agent_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_pitched_ptr_s(Structure):
    pass

struct_hsa_pitched_ptr_s._pack_ = 1 # source:False
struct_hsa_pitched_ptr_s._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('pitch', ctypes.c_uint64),
    ('slice', ctypes.c_uint64),
]

hsa_pitched_ptr_t = struct_hsa_pitched_ptr_s

# values for enumeration 'hsa_amd_copy_direction_t'
hsa_amd_copy_direction_t__enumvalues = {
    0: 'hsaHostToHost',
    1: 'hsaHostToDevice',
    2: 'hsaDeviceToHost',
    3: 'hsaDeviceToDevice',
}
hsaHostToHost = 0
hsaHostToDevice = 1
hsaDeviceToHost = 2
hsaDeviceToDevice = 3
hsa_amd_copy_direction_t = ctypes.c_uint32 # enum
try:
    hsa_amd_memory_async_copy_rect = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy_rect
    hsa_amd_memory_async_copy_rect.restype = hsa_status_t
    hsa_amd_memory_async_copy_rect.argtypes = [ctypes.POINTER(struct_hsa_pitched_ptr_s), ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_pitched_ptr_s), ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s), hsa_agent_t, hsa_amd_copy_direction_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_memory_pool_access_t'
hsa_amd_memory_pool_access_t__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED',
    1: 'HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT',
    2: 'HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT',
}
HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0
HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1
HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2
hsa_amd_memory_pool_access_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_link_info_type_t'
hsa_amd_link_info_type_t__enumvalues = {
    0: 'HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT',
    1: 'HSA_AMD_LINK_INFO_TYPE_QPI',
    2: 'HSA_AMD_LINK_INFO_TYPE_PCIE',
    3: 'HSA_AMD_LINK_INFO_TYPE_INFINBAND',
    4: 'HSA_AMD_LINK_INFO_TYPE_XGMI',
}
HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = 0
HSA_AMD_LINK_INFO_TYPE_QPI = 1
HSA_AMD_LINK_INFO_TYPE_PCIE = 2
HSA_AMD_LINK_INFO_TYPE_INFINBAND = 3
HSA_AMD_LINK_INFO_TYPE_XGMI = 4
hsa_amd_link_info_type_t = ctypes.c_uint32 # enum
class struct_hsa_amd_memory_pool_link_info_s(Structure):
    pass

struct_hsa_amd_memory_pool_link_info_s._pack_ = 1 # source:False
struct_hsa_amd_memory_pool_link_info_s._fields_ = [
    ('min_latency', ctypes.c_uint32),
    ('max_latency', ctypes.c_uint32),
    ('min_bandwidth', ctypes.c_uint32),
    ('max_bandwidth', ctypes.c_uint32),
    ('atomic_support_32bit', ctypes.c_bool),
    ('atomic_support_64bit', ctypes.c_bool),
    ('coherent_support', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('link_type', hsa_amd_link_info_type_t),
    ('numa_distance', ctypes.c_uint32),
]

hsa_amd_memory_pool_link_info_t = struct_hsa_amd_memory_pool_link_info_s

# values for enumeration 'hsa_amd_agent_memory_pool_info_t'
hsa_amd_agent_memory_pool_info_t__enumvalues = {
    0: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS',
    1: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS',
    2: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO',
}
HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0
HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = 1
HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = 2
hsa_amd_agent_memory_pool_info_t = ctypes.c_uint32 # enum
try:
    hsa_amd_agent_memory_pool_get_info = _libraries['libhsa-runtime64.so'].hsa_amd_agent_memory_pool_get_info
    hsa_amd_agent_memory_pool_get_info.restype = hsa_status_t
    hsa_amd_agent_memory_pool_get_info.argtypes = [hsa_agent_t, hsa_amd_memory_pool_t, hsa_amd_agent_memory_pool_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_agents_allow_access = _libraries['libhsa-runtime64.so'].hsa_amd_agents_allow_access
    hsa_amd_agents_allow_access.restype = hsa_status_t
    hsa_amd_agents_allow_access.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_can_migrate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_can_migrate
    hsa_amd_memory_pool_can_migrate.restype = hsa_status_t
    hsa_amd_memory_pool_can_migrate.argtypes = [hsa_amd_memory_pool_t, hsa_amd_memory_pool_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_amd_memory_migrate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_migrate
    hsa_amd_memory_migrate.restype = hsa_status_t
    hsa_amd_memory_migrate.argtypes = [ctypes.POINTER(None), hsa_amd_memory_pool_t, uint32_t]
except AttributeError:
    pass
try:
    hsa_amd_memory_lock = _libraries['libhsa-runtime64.so'].hsa_amd_memory_lock
    hsa_amd_memory_lock.restype = hsa_status_t
    hsa_amd_memory_lock.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_lock_to_pool = _libraries['libhsa-runtime64.so'].hsa_amd_memory_lock_to_pool
    hsa_amd_memory_lock_to_pool.restype = hsa_status_t
    hsa_amd_memory_lock_to_pool.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, hsa_amd_memory_pool_t, uint32_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_unlock = _libraries['libhsa-runtime64.so'].hsa_amd_memory_unlock
    hsa_amd_memory_unlock.restype = hsa_status_t
    hsa_amd_memory_unlock.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_fill = _libraries['libhsa-runtime64.so'].hsa_amd_memory_fill
    hsa_amd_memory_fill.restype = hsa_status_t
    hsa_amd_memory_fill.argtypes = [ctypes.POINTER(None), uint32_t, size_t]
except AttributeError:
    pass
try:
    hsa_amd_interop_map_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_interop_map_buffer
    hsa_amd_interop_map_buffer.restype = hsa_status_t
    hsa_amd_interop_map_buffer.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, uint32_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_interop_unmap_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_interop_unmap_buffer
    hsa_amd_interop_unmap_buffer.restype = hsa_status_t
    hsa_amd_interop_unmap_buffer.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_amd_image_descriptor_s(Structure):
    pass

struct_hsa_amd_image_descriptor_s._pack_ = 1 # source:False
struct_hsa_amd_image_descriptor_s._fields_ = [
    ('version', ctypes.c_uint32),
    ('deviceID', ctypes.c_uint32),
    ('data', ctypes.c_uint32 * 1),
]

hsa_amd_image_descriptor_t = struct_hsa_amd_image_descriptor_s
try:
    hsa_amd_image_create = _libraries['libhsa-runtime64.so'].hsa_amd_image_create
    hsa_amd_image_create.restype = hsa_status_t
    hsa_amd_image_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(struct_hsa_amd_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_pointer_type_t'
hsa_amd_pointer_type_t__enumvalues = {
    0: 'HSA_EXT_POINTER_TYPE_UNKNOWN',
    1: 'HSA_EXT_POINTER_TYPE_HSA',
    2: 'HSA_EXT_POINTER_TYPE_LOCKED',
    3: 'HSA_EXT_POINTER_TYPE_GRAPHICS',
    4: 'HSA_EXT_POINTER_TYPE_IPC',
}
HSA_EXT_POINTER_TYPE_UNKNOWN = 0
HSA_EXT_POINTER_TYPE_HSA = 1
HSA_EXT_POINTER_TYPE_LOCKED = 2
HSA_EXT_POINTER_TYPE_GRAPHICS = 3
HSA_EXT_POINTER_TYPE_IPC = 4
hsa_amd_pointer_type_t = ctypes.c_uint32 # enum
class struct_hsa_amd_pointer_info_s(Structure):
    pass

struct_hsa_amd_pointer_info_s._pack_ = 1 # source:False
struct_hsa_amd_pointer_info_s._fields_ = [
    ('size', ctypes.c_uint32),
    ('type', hsa_amd_pointer_type_t),
    ('agentBaseAddress', ctypes.POINTER(None)),
    ('hostBaseAddress', ctypes.POINTER(None)),
    ('sizeInBytes', ctypes.c_uint64),
    ('userData', ctypes.POINTER(None)),
    ('agentOwner', hsa_agent_t),
    ('global_flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hsa_amd_pointer_info_t = struct_hsa_amd_pointer_info_s
try:
    hsa_amd_pointer_info = _libraries['libhsa-runtime64.so'].hsa_amd_pointer_info
    hsa_amd_pointer_info.restype = hsa_status_t
    hsa_amd_pointer_info.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_hsa_amd_pointer_info_s), ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct_hsa_agent_s))]
except AttributeError:
    pass
try:
    hsa_amd_pointer_info_set_userdata = _libraries['libhsa-runtime64.so'].hsa_amd_pointer_info_set_userdata
    hsa_amd_pointer_info_set_userdata.restype = hsa_status_t
    hsa_amd_pointer_info_set_userdata.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_amd_ipc_memory_s(Structure):
    pass

struct_hsa_amd_ipc_memory_s._pack_ = 1 # source:False
struct_hsa_amd_ipc_memory_s._fields_ = [
    ('handle', ctypes.c_uint32 * 8),
]

hsa_amd_ipc_memory_t = struct_hsa_amd_ipc_memory_s
try:
    hsa_amd_ipc_memory_create = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_create
    hsa_amd_ipc_memory_create.restype = hsa_status_t
    hsa_amd_ipc_memory_create.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_ipc_memory_s)]
except AttributeError:
    pass
try:
    hsa_amd_ipc_memory_attach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_attach
    hsa_amd_ipc_memory_attach.restype = hsa_status_t
    hsa_amd_ipc_memory_attach.argtypes = [ctypes.POINTER(struct_hsa_amd_ipc_memory_s), size_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_ipc_memory_detach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_detach
    hsa_amd_ipc_memory_detach.restype = hsa_status_t
    hsa_amd_ipc_memory_detach.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
hsa_amd_ipc_signal_t = struct_hsa_amd_ipc_memory_s
try:
    hsa_amd_ipc_signal_create = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_signal_create
    hsa_amd_ipc_signal_create.restype = hsa_status_t
    hsa_amd_ipc_signal_create.argtypes = [hsa_signal_t, ctypes.POINTER(struct_hsa_amd_ipc_memory_s)]
except AttributeError:
    pass
try:
    hsa_amd_ipc_signal_attach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_signal_attach
    hsa_amd_ipc_signal_attach.restype = hsa_status_t
    hsa_amd_ipc_signal_attach.argtypes = [ctypes.POINTER(struct_hsa_amd_ipc_memory_s), ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_event_type_s'
hsa_amd_event_type_s__enumvalues = {
    0: 'HSA_AMD_GPU_MEMORY_FAULT_EVENT',
    1: 'HSA_AMD_GPU_HW_EXCEPTION_EVENT',
}
HSA_AMD_GPU_MEMORY_FAULT_EVENT = 0
HSA_AMD_GPU_HW_EXCEPTION_EVENT = 1
hsa_amd_event_type_s = ctypes.c_uint32 # enum
hsa_amd_event_type_t = hsa_amd_event_type_s
hsa_amd_event_type_t__enumvalues = hsa_amd_event_type_s__enumvalues

# values for enumeration 'hsa_amd_memory_fault_reason_t'
hsa_amd_memory_fault_reason_t__enumvalues = {
    1: 'HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT',
    2: 'HSA_AMD_MEMORY_FAULT_READ_ONLY',
    4: 'HSA_AMD_MEMORY_FAULT_NX',
    8: 'HSA_AMD_MEMORY_FAULT_HOST_ONLY',
    16: 'HSA_AMD_MEMORY_FAULT_DRAMECC',
    32: 'HSA_AMD_MEMORY_FAULT_IMPRECISE',
    64: 'HSA_AMD_MEMORY_FAULT_SRAMECC',
    -2147483648: 'HSA_AMD_MEMORY_FAULT_HANG',
}
HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = 1
HSA_AMD_MEMORY_FAULT_READ_ONLY = 2
HSA_AMD_MEMORY_FAULT_NX = 4
HSA_AMD_MEMORY_FAULT_HOST_ONLY = 8
HSA_AMD_MEMORY_FAULT_DRAMECC = 16
HSA_AMD_MEMORY_FAULT_IMPRECISE = 32
HSA_AMD_MEMORY_FAULT_SRAMECC = 64
HSA_AMD_MEMORY_FAULT_HANG = -2147483648
hsa_amd_memory_fault_reason_t = ctypes.c_int32 # enum
class struct_hsa_amd_gpu_memory_fault_info_s(Structure):
    pass

struct_hsa_amd_gpu_memory_fault_info_s._pack_ = 1 # source:False
struct_hsa_amd_gpu_memory_fault_info_s._fields_ = [
    ('agent', hsa_agent_t),
    ('virtual_address', ctypes.c_uint64),
    ('fault_reason_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hsa_amd_gpu_memory_fault_info_t = struct_hsa_amd_gpu_memory_fault_info_s

# values for enumeration 'hsa_amd_hw_exception_reset_type_t'
hsa_amd_hw_exception_reset_type_t__enumvalues = {
    1: 'HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER',
}
HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER = 1
hsa_amd_hw_exception_reset_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_hw_exception_reset_cause_t'
hsa_amd_hw_exception_reset_cause_t__enumvalues = {
    1: 'HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG',
    2: 'HSA_AMD_HW_EXCEPTION_CAUSE_ECC',
}
HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG = 1
HSA_AMD_HW_EXCEPTION_CAUSE_ECC = 2
hsa_amd_hw_exception_reset_cause_t = ctypes.c_uint32 # enum
class struct_hsa_amd_gpu_hw_exception_info_s(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('agent', hsa_agent_t),
    ('reset_type', hsa_amd_hw_exception_reset_type_t),
    ('reset_cause', hsa_amd_hw_exception_reset_cause_t),
     ]

hsa_amd_gpu_hw_exception_info_t = struct_hsa_amd_gpu_hw_exception_info_s
class struct_hsa_amd_event_s(Structure):
    pass

class union_union_hsa_ext_amd_h_2329(Union):
    pass

union_union_hsa_ext_amd_h_2329._pack_ = 1 # source:False
union_union_hsa_ext_amd_h_2329._fields_ = [
    ('memory_fault', hsa_amd_gpu_memory_fault_info_t),
    ('hw_exception', hsa_amd_gpu_hw_exception_info_t),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_hsa_amd_event_s._pack_ = 1 # source:False
struct_hsa_amd_event_s._anonymous_ = ('_0',)
struct_hsa_amd_event_s._fields_ = [
    ('event_type', hsa_amd_event_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_union_hsa_ext_amd_h_2329),
]

hsa_amd_event_t = struct_hsa_amd_event_s
hsa_amd_system_event_callback_t = ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(struct_hsa_amd_event_s), ctypes.POINTER(None))
try:
    hsa_amd_register_system_event_handler = _libraries['libhsa-runtime64.so'].hsa_amd_register_system_event_handler
    hsa_amd_register_system_event_handler.restype = hsa_status_t
    hsa_amd_register_system_event_handler.argtypes = [hsa_amd_system_event_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_queue_priority_s'
hsa_amd_queue_priority_s__enumvalues = {
    0: 'HSA_AMD_QUEUE_PRIORITY_LOW',
    1: 'HSA_AMD_QUEUE_PRIORITY_NORMAL',
    2: 'HSA_AMD_QUEUE_PRIORITY_HIGH',
}
HSA_AMD_QUEUE_PRIORITY_LOW = 0
HSA_AMD_QUEUE_PRIORITY_NORMAL = 1
HSA_AMD_QUEUE_PRIORITY_HIGH = 2
hsa_amd_queue_priority_s = ctypes.c_uint32 # enum
hsa_amd_queue_priority_t = hsa_amd_queue_priority_s
hsa_amd_queue_priority_t__enumvalues = hsa_amd_queue_priority_s__enumvalues
try:
    hsa_amd_queue_set_priority = _libraries['libhsa-runtime64.so'].hsa_amd_queue_set_priority
    hsa_amd_queue_set_priority.restype = hsa_status_t
    hsa_amd_queue_set_priority.argtypes = [ctypes.POINTER(struct_hsa_queue_s), hsa_amd_queue_priority_t]
except AttributeError:
    pass
hsa_amd_deallocation_callback_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None))
try:
    hsa_amd_register_deallocation_callback = _libraries['libhsa-runtime64.so'].hsa_amd_register_deallocation_callback
    hsa_amd_register_deallocation_callback.restype = hsa_status_t
    hsa_amd_register_deallocation_callback.argtypes = [ctypes.POINTER(None), hsa_amd_deallocation_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_deregister_deallocation_callback = _libraries['libhsa-runtime64.so'].hsa_amd_deregister_deallocation_callback
    hsa_amd_deregister_deallocation_callback.restype = hsa_status_t
    hsa_amd_deregister_deallocation_callback.argtypes = [ctypes.POINTER(None), hsa_amd_deallocation_callback_t]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_svm_model_s'
hsa_amd_svm_model_s__enumvalues = {
    0: 'HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED',
    1: 'HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED',
    2: 'HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE',
}
HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED = 0
HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED = 1
HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE = 2
hsa_amd_svm_model_s = ctypes.c_uint32 # enum
hsa_amd_svm_model_t = hsa_amd_svm_model_s
hsa_amd_svm_model_t__enumvalues = hsa_amd_svm_model_s__enumvalues

# values for enumeration 'hsa_amd_svm_attribute_s'
hsa_amd_svm_attribute_s__enumvalues = {
    0: 'HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG',
    1: 'HSA_AMD_SVM_ATTRIB_READ_ONLY',
    2: 'HSA_AMD_SVM_ATTRIB_HIVE_LOCAL',
    3: 'HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY',
    4: 'HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION',
    5: 'HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION',
    6: 'HSA_AMD_SVM_ATTRIB_READ_MOSTLY',
    7: 'HSA_AMD_SVM_ATTRIB_GPU_EXEC',
    512: 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE',
    513: 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE',
    514: 'HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS',
    515: 'HSA_AMD_SVM_ATTRIB_ACCESS_QUERY',
}
HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = 0
HSA_AMD_SVM_ATTRIB_READ_ONLY = 1
HSA_AMD_SVM_ATTRIB_HIVE_LOCAL = 2
HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY = 3
HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION = 4
HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION = 5
HSA_AMD_SVM_ATTRIB_READ_MOSTLY = 6
HSA_AMD_SVM_ATTRIB_GPU_EXEC = 7
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = 512
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE = 513
HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS = 514
HSA_AMD_SVM_ATTRIB_ACCESS_QUERY = 515
hsa_amd_svm_attribute_s = ctypes.c_uint32 # enum
hsa_amd_svm_attribute_t = hsa_amd_svm_attribute_s
hsa_amd_svm_attribute_t__enumvalues = hsa_amd_svm_attribute_s__enumvalues
class struct_hsa_amd_svm_attribute_pair_s(Structure):
    pass

struct_hsa_amd_svm_attribute_pair_s._pack_ = 1 # source:False
struct_hsa_amd_svm_attribute_pair_s._fields_ = [
    ('attribute', ctypes.c_uint64),
    ('value', ctypes.c_uint64),
]

hsa_amd_svm_attribute_pair_t = struct_hsa_amd_svm_attribute_pair_s
try:
    hsa_amd_svm_attributes_set = _libraries['libhsa-runtime64.so'].hsa_amd_svm_attributes_set
    hsa_amd_svm_attributes_set.restype = hsa_status_t
    hsa_amd_svm_attributes_set.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_svm_attribute_pair_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_svm_attributes_get = _libraries['libhsa-runtime64.so'].hsa_amd_svm_attributes_get
    hsa_amd_svm_attributes_get.restype = hsa_status_t
    hsa_amd_svm_attributes_get.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_svm_attribute_pair_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_svm_prefetch_async = _libraries['libhsa-runtime64.so'].hsa_amd_svm_prefetch_async
    hsa_amd_svm_prefetch_async.restype = hsa_status_t
    hsa_amd_svm_prefetch_async.argtypes = [ctypes.POINTER(None), size_t, hsa_agent_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_acquire = _libraries['libhsa-runtime64.so'].hsa_amd_spm_acquire
    hsa_amd_spm_acquire.restype = hsa_status_t
    hsa_amd_spm_acquire.argtypes = [hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_release = _libraries['libhsa-runtime64.so'].hsa_amd_spm_release
    hsa_amd_spm_release.restype = hsa_status_t
    hsa_amd_spm_release.argtypes = [hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_set_dest_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_spm_set_dest_buffer
    hsa_amd_spm_set_dest_buffer.restype = hsa_status_t
    hsa_amd_spm_set_dest_buffer.argtypes = [hsa_agent_t, size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_amd_portable_export_dmabuf = _libraries['libhsa-runtime64.so'].hsa_amd_portable_export_dmabuf
    hsa_amd_portable_export_dmabuf.restype = hsa_status_t
    hsa_amd_portable_export_dmabuf.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsa_amd_portable_close_dmabuf = _libraries['libhsa-runtime64.so'].hsa_amd_portable_close_dmabuf
    hsa_amd_portable_close_dmabuf.restype = hsa_status_t
    hsa_amd_portable_close_dmabuf.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    hsa_amd_vmem_address_reserve = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_address_reserve
    hsa_amd_vmem_address_reserve.restype = hsa_status_t
    hsa_amd_vmem_address_reserve.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_address_free = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_address_free
    hsa_amd_vmem_address_free.restype = hsa_status_t
    hsa_amd_vmem_address_free.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_amd_vmem_alloc_handle_s(Structure):
    pass

struct_hsa_amd_vmem_alloc_handle_s._pack_ = 1 # source:False
struct_hsa_amd_vmem_alloc_handle_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_amd_vmem_alloc_handle_t = struct_hsa_amd_vmem_alloc_handle_s

# values for enumeration 'hsa_amd_memory_type_t'
hsa_amd_memory_type_t__enumvalues = {
    0: 'MEMORY_TYPE_NONE',
    1: 'MEMORY_TYPE_PINNED',
}
MEMORY_TYPE_NONE = 0
MEMORY_TYPE_PINNED = 1
hsa_amd_memory_type_t = ctypes.c_uint32 # enum
try:
    hsa_amd_vmem_handle_create = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_handle_create
    hsa_amd_vmem_handle_create.restype = hsa_status_t
    hsa_amd_vmem_handle_create.argtypes = [hsa_amd_memory_pool_t, size_t, hsa_amd_memory_type_t, uint64_t, ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_handle_release = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_handle_release
    hsa_amd_vmem_handle_release.restype = hsa_status_t
    hsa_amd_vmem_handle_release.argtypes = [hsa_amd_vmem_alloc_handle_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_map = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_map
    hsa_amd_vmem_map.restype = hsa_status_t
    hsa_amd_vmem_map.argtypes = [ctypes.POINTER(None), size_t, size_t, hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_unmap = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_unmap
    hsa_amd_vmem_unmap.restype = hsa_status_t
    hsa_amd_vmem_unmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_amd_memory_access_desc_s(Structure):
    pass

struct_hsa_amd_memory_access_desc_s._pack_ = 1 # source:False
struct_hsa_amd_memory_access_desc_s._fields_ = [
    ('permissions', hsa_access_permission_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('agent_handle', hsa_agent_t),
]

hsa_amd_memory_access_desc_t = struct_hsa_amd_memory_access_desc_s
try:
    hsa_amd_vmem_set_access = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_set_access
    hsa_amd_vmem_set_access.restype = hsa_status_t
    hsa_amd_vmem_set_access.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_memory_access_desc_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_get_access = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_get_access
    hsa_amd_vmem_get_access.restype = hsa_status_t
    hsa_amd_vmem_get_access.argtypes = [ctypes.POINTER(None), ctypes.POINTER(hsa_access_permission_t), hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_export_shareable_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_export_shareable_handle
    hsa_amd_vmem_export_shareable_handle.restype = hsa_status_t
    hsa_amd_vmem_export_shareable_handle.argtypes = [ctypes.POINTER(ctypes.c_int32), hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_import_shareable_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_import_shareable_handle
    hsa_amd_vmem_import_shareable_handle.restype = hsa_status_t
    hsa_amd_vmem_import_shareable_handle.argtypes = [ctypes.c_int32, ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_retain_alloc_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_retain_alloc_handle
    hsa_amd_vmem_retain_alloc_handle.restype = hsa_status_t
    hsa_amd_vmem_retain_alloc_handle.argtypes = [ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_get_alloc_properties_from_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_get_alloc_properties_from_handle
    hsa_amd_vmem_get_alloc_properties_from_handle.restype = hsa_status_t
    hsa_amd_vmem_get_alloc_properties_from_handle.argtypes = [hsa_amd_vmem_alloc_handle_t, ctypes.POINTER(struct_hsa_amd_memory_pool_s), ctypes.POINTER(hsa_amd_memory_type_t)]
except AttributeError:
    pass
class struct_BrigModuleHeader(Structure):
    pass

BrigModule_t = ctypes.POINTER(struct_BrigModuleHeader)

# values for enumeration 'enum_hsa_ext_finalize_h_69'
enum_hsa_ext_finalize_h_69__enumvalues = {
    8192: 'HSA_EXT_STATUS_ERROR_INVALID_PROGRAM',
    8193: 'HSA_EXT_STATUS_ERROR_INVALID_MODULE',
    8194: 'HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE',
    8195: 'HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED',
    8196: 'HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH',
    8197: 'HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED',
    8198: 'HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH',
}
HSA_EXT_STATUS_ERROR_INVALID_PROGRAM = 8192
HSA_EXT_STATUS_ERROR_INVALID_MODULE = 8193
HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE = 8194
HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED = 8195
HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH = 8196
HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED = 8197
HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH = 8198
enum_hsa_ext_finalize_h_69 = ctypes.c_uint32 # enum
hsa_ext_module_t = ctypes.POINTER(struct_BrigModuleHeader)
class struct_hsa_ext_program_s(Structure):
    pass

struct_hsa_ext_program_s._pack_ = 1 # source:False
struct_hsa_ext_program_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_program_t = struct_hsa_ext_program_s
try:
    hsa_ext_program_create = _libraries['libhsa-runtime64.so'].hsa_ext_program_create
    hsa_ext_program_create.restype = hsa_status_t
    hsa_ext_program_create.argtypes = [hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_ext_program_s)]
except AttributeError:
    pass
try:
    hsa_ext_program_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_program_destroy
    hsa_ext_program_destroy.restype = hsa_status_t
    hsa_ext_program_destroy.argtypes = [hsa_ext_program_t]
except AttributeError:
    pass
try:
    hsa_ext_program_add_module = _libraries['libhsa-runtime64.so'].hsa_ext_program_add_module
    hsa_ext_program_add_module.restype = hsa_status_t
    hsa_ext_program_add_module.argtypes = [hsa_ext_program_t, hsa_ext_module_t]
except AttributeError:
    pass
try:
    hsa_ext_program_iterate_modules = _libraries['libhsa-runtime64.so'].hsa_ext_program_iterate_modules
    hsa_ext_program_iterate_modules.restype = hsa_status_t
    hsa_ext_program_iterate_modules.argtypes = [hsa_ext_program_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_ext_program_info_t'
hsa_ext_program_info_t__enumvalues = {
    0: 'HSA_EXT_PROGRAM_INFO_MACHINE_MODEL',
    1: 'HSA_EXT_PROGRAM_INFO_PROFILE',
    2: 'HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_EXT_PROGRAM_INFO_MACHINE_MODEL = 0
HSA_EXT_PROGRAM_INFO_PROFILE = 1
HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 2
hsa_ext_program_info_t = ctypes.c_uint32 # enum
try:
    hsa_ext_program_get_info = _libraries['libhsa-runtime64.so'].hsa_ext_program_get_info
    hsa_ext_program_get_info.restype = hsa_status_t
    hsa_ext_program_get_info.argtypes = [hsa_ext_program_t, hsa_ext_program_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_ext_finalizer_call_convention_t'
hsa_ext_finalizer_call_convention_t__enumvalues = {
    -1: 'HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO',
}
HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO = -1
hsa_ext_finalizer_call_convention_t = ctypes.c_int32 # enum
class struct_hsa_ext_control_directives_s(Structure):
    pass

struct_hsa_ext_control_directives_s._pack_ = 1 # source:False
struct_hsa_ext_control_directives_s._fields_ = [
    ('control_directives_mask', ctypes.c_uint64),
    ('break_exceptions_mask', ctypes.c_uint16),
    ('detect_exceptions_mask', ctypes.c_uint16),
    ('max_dynamic_group_size', ctypes.c_uint32),
    ('max_flat_grid_size', ctypes.c_uint64),
    ('max_flat_workgroup_size', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('required_grid_size', ctypes.c_uint64 * 3),
    ('required_workgroup_size', hsa_dim3_t),
    ('required_dim', ctypes.c_ubyte),
    ('reserved2', ctypes.c_ubyte * 75),
]

hsa_ext_control_directives_t = struct_hsa_ext_control_directives_s
try:
    hsa_ext_program_finalize = _libraries['libhsa-runtime64.so'].hsa_ext_program_finalize
    hsa_ext_program_finalize.restype = hsa_status_t
    hsa_ext_program_finalize.argtypes = [hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(struct_hsa_code_object_s)]
except AttributeError:
    pass
class struct_hsa_ext_finalizer_1_00_pfn_s(Structure):
    pass

struct_hsa_ext_finalizer_1_00_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_finalizer_1_00_pfn_s._fields_ = [
    ('hsa_ext_program_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_ext_program_s))),
    ('hsa_ext_program_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s)),
    ('hsa_ext_program_add_module', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader))),
    ('hsa_ext_program_iterate_modules', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader), ctypes.POINTER(None)), ctypes.POINTER(None))),
    ('hsa_ext_program_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, hsa_ext_program_info_t, ctypes.POINTER(None))),
    ('hsa_ext_program_finalize', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, struct_hsa_isa_s, ctypes.c_int32, struct_hsa_ext_control_directives_s, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(struct_hsa_code_object_s))),
]

hsa_ext_finalizer_1_00_pfn_t = struct_hsa_ext_finalizer_1_00_pfn_s
__all__ = \
    ['BrigModule_t', 'HSA_ACCESS_PERMISSION_NONE',
    'HSA_ACCESS_PERMISSION_RO', 'HSA_ACCESS_PERMISSION_RW',
    'HSA_ACCESS_PERMISSION_WO', 'HSA_AGENT_FEATURE_AGENT_DISPATCH',
    'HSA_AGENT_FEATURE_KERNEL_DISPATCH',
    'HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_AGENT_INFO_CACHE_SIZE',
    'HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_AGENT_INFO_DEVICE', 'HSA_AGENT_INFO_EXTENSIONS',
    'HSA_AGENT_INFO_FAST_F16_OPERATION',
    'HSA_AGENT_INFO_FBARRIER_MAX_SIZE', 'HSA_AGENT_INFO_FEATURE',
    'HSA_AGENT_INFO_GRID_MAX_DIM', 'HSA_AGENT_INFO_GRID_MAX_SIZE',
    'HSA_AGENT_INFO_ISA', 'HSA_AGENT_INFO_LAST',
    'HSA_AGENT_INFO_MACHINE_MODEL', 'HSA_AGENT_INFO_NAME',
    'HSA_AGENT_INFO_NODE', 'HSA_AGENT_INFO_PROFILE',
    'HSA_AGENT_INFO_QUEUES_MAX', 'HSA_AGENT_INFO_QUEUE_MAX_SIZE',
    'HSA_AGENT_INFO_QUEUE_MIN_SIZE', 'HSA_AGENT_INFO_QUEUE_TYPE',
    'HSA_AGENT_INFO_VENDOR_NAME', 'HSA_AGENT_INFO_VERSION_MAJOR',
    'HSA_AGENT_INFO_VERSION_MINOR', 'HSA_AGENT_INFO_WAVEFRONT_SIZE',
    'HSA_AGENT_INFO_WORKGROUP_MAX_DIM',
    'HSA_AGENT_INFO_WORKGROUP_MAX_SIZE',
    'HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID',
    'HSA_AMD_AGENT_INFO_ASIC_REVISION', 'HSA_AMD_AGENT_INFO_BDFID',
    'HSA_AMD_AGENT_INFO_CACHELINE_SIZE', 'HSA_AMD_AGENT_INFO_CHIP_ID',
    'HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT',
    'HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT',
    'HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES',
    'HSA_AMD_AGENT_INFO_DOMAIN', 'HSA_AMD_AGENT_INFO_DRIVER_NODE_ID',
    'HSA_AMD_AGENT_INFO_DRIVER_UID', 'HSA_AMD_AGENT_INFO_HDP_FLUSH',
    'HSA_AMD_AGENT_INFO_IOMMU_SUPPORT',
    'HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS',
    'HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY',
    'HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU',
    'HSA_AMD_AGENT_INFO_MEMORY_AVAIL',
    'HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY',
    'HSA_AMD_AGENT_INFO_MEMORY_WIDTH',
    'HSA_AMD_AGENT_INFO_NEAREST_CPU',
    'HSA_AMD_AGENT_INFO_NUM_SDMA_ENG',
    'HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG',
    'HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE',
    'HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES',
    'HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU',
    'HSA_AMD_AGENT_INFO_NUM_XCC', 'HSA_AMD_AGENT_INFO_PRODUCT_NAME',
    'HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION',
    'HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS',
    'HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY',
    'HSA_AMD_AGENT_INFO_UCODE_VERSION', 'HSA_AMD_AGENT_INFO_UUID',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS',
    'HSA_AMD_COHERENCY_TYPE_COHERENT',
    'HSA_AMD_COHERENCY_TYPE_NONCOHERENT', 'HSA_AMD_FIRST_EXTENSION',
    'HSA_AMD_GPU_HW_EXCEPTION_EVENT',
    'HSA_AMD_GPU_MEMORY_FAULT_EVENT',
    'HSA_AMD_HW_EXCEPTION_CAUSE_ECC',
    'HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG',
    'HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER', 'HSA_AMD_LAST_EXTENSION',
    'HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT',
    'HSA_AMD_LINK_INFO_TYPE_INFINBAND', 'HSA_AMD_LINK_INFO_TYPE_PCIE',
    'HSA_AMD_LINK_INFO_TYPE_QPI', 'HSA_AMD_LINK_INFO_TYPE_XGMI',
    'HSA_AMD_MEMORY_FAULT_DRAMECC', 'HSA_AMD_MEMORY_FAULT_HANG',
    'HSA_AMD_MEMORY_FAULT_HOST_ONLY',
    'HSA_AMD_MEMORY_FAULT_IMPRECISE', 'HSA_AMD_MEMORY_FAULT_NX',
    'HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT',
    'HSA_AMD_MEMORY_FAULT_READ_ONLY', 'HSA_AMD_MEMORY_FAULT_SRAMECC',
    'HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT',
    'HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT',
    'HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT',
    'HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL',
    'HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE',
    'HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS',
    'HSA_AMD_MEMORY_POOL_INFO_LOCATION',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE',
    'HSA_AMD_MEMORY_POOL_INFO_SEGMENT',
    'HSA_AMD_MEMORY_POOL_INFO_SIZE',
    'HSA_AMD_MEMORY_POOL_LOCATION_CPU',
    'HSA_AMD_MEMORY_POOL_LOCATION_GPU',
    'HSA_AMD_MEMORY_POOL_PCIE_FLAG',
    'HSA_AMD_MEMORY_POOL_STANDARD_FLAG',
    'HSA_AMD_PACKET_TYPE_BARRIER_VALUE',
    'HSA_AMD_QUEUE_PRIORITY_HIGH', 'HSA_AMD_QUEUE_PRIORITY_LOW',
    'HSA_AMD_QUEUE_PRIORITY_NORMAL', 'HSA_AMD_REGION_INFO_BASE',
    'HSA_AMD_REGION_INFO_BUS_WIDTH',
    'HSA_AMD_REGION_INFO_HOST_ACCESSIBLE',
    'HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY',
    'HSA_AMD_SDMA_ENGINE_0', 'HSA_AMD_SDMA_ENGINE_1',
    'HSA_AMD_SDMA_ENGINE_10', 'HSA_AMD_SDMA_ENGINE_11',
    'HSA_AMD_SDMA_ENGINE_12', 'HSA_AMD_SDMA_ENGINE_13',
    'HSA_AMD_SDMA_ENGINE_14', 'HSA_AMD_SDMA_ENGINE_15',
    'HSA_AMD_SDMA_ENGINE_2', 'HSA_AMD_SDMA_ENGINE_3',
    'HSA_AMD_SDMA_ENGINE_4', 'HSA_AMD_SDMA_ENGINE_5',
    'HSA_AMD_SDMA_ENGINE_6', 'HSA_AMD_SDMA_ENGINE_7',
    'HSA_AMD_SDMA_ENGINE_8', 'HSA_AMD_SDMA_ENGINE_9',
    'HSA_AMD_SEGMENT_GLOBAL', 'HSA_AMD_SEGMENT_GROUP',
    'HSA_AMD_SEGMENT_PRIVATE', 'HSA_AMD_SEGMENT_READONLY',
    'HSA_AMD_SIGNAL_AMD_GPU_ONLY', 'HSA_AMD_SIGNAL_IPC',
    'HSA_AMD_SVM_ATTRIB_ACCESS_QUERY',
    'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE',
    'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE',
    'HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS',
    'HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG', 'HSA_AMD_SVM_ATTRIB_GPU_EXEC',
    'HSA_AMD_SVM_ATTRIB_HIVE_LOCAL',
    'HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY',
    'HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION',
    'HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION',
    'HSA_AMD_SVM_ATTRIB_READ_MOSTLY', 'HSA_AMD_SVM_ATTRIB_READ_ONLY',
    'HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE',
    'HSA_AMD_SYSTEM_INFO_BUILD_VERSION',
    'HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED',
    'HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT',
    'HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_XNACK_ENABLED', 'HSA_CACHE_INFO_LEVEL',
    'HSA_CACHE_INFO_NAME', 'HSA_CACHE_INFO_NAME_LENGTH',
    'HSA_CACHE_INFO_SIZE',
    'HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_CODE_OBJECT_INFO_ISA', 'HSA_CODE_OBJECT_INFO_MACHINE_MODEL',
    'HSA_CODE_OBJECT_INFO_PROFILE', 'HSA_CODE_OBJECT_INFO_TYPE',
    'HSA_CODE_OBJECT_INFO_VERSION', 'HSA_CODE_OBJECT_TYPE_PROGRAM',
    'HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
    'HSA_CODE_SYMBOL_INFO_IS_DEFINITION',
    'HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    'HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    'HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_LINKAGE',
    'HSA_CODE_SYMBOL_INFO_MODULE_NAME',
    'HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    'HSA_CODE_SYMBOL_INFO_NAME', 'HSA_CODE_SYMBOL_INFO_NAME_LENGTH',
    'HSA_CODE_SYMBOL_INFO_TYPE',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO', 'HSA_DEVICE_TYPE_CPU',
    'HSA_DEVICE_TYPE_DSP', 'HSA_DEVICE_TYPE_GPU',
    'HSA_ENDIANNESS_BIG', 'HSA_ENDIANNESS_LITTLE',
    'HSA_EXCEPTION_POLICY_BREAK', 'HSA_EXCEPTION_POLICY_DETECT',
    'HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_EXECUTABLE_INFO_PROFILE', 'HSA_EXECUTABLE_INFO_STATE',
    'HSA_EXECUTABLE_STATE_FROZEN', 'HSA_EXECUTABLE_STATE_UNFROZEN',
    'HSA_EXECUTABLE_SYMBOL_INFO_AGENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
    'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT',
    'HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE',
    'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME',
    'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    'HSA_EXECUTABLE_SYMBOL_INFO_NAME',
    'HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH',
    'HSA_EXECUTABLE_SYMBOL_INFO_TYPE',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE',
    'HSA_EXTENSION_AMD_AQLPROFILE', 'HSA_EXTENSION_AMD_LOADER',
    'HSA_EXTENSION_AMD_PROFILER', 'HSA_EXTENSION_FINALIZER',
    'HSA_EXTENSION_IMAGES', 'HSA_EXTENSION_PERFORMANCE_COUNTERS',
    'HSA_EXTENSION_PROFILING_EVENTS', 'HSA_EXTENSION_STD_LAST',
    'HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS',
    'HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT',
    'HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES',
    'HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES',
    'HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS',
    'HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO',
    'HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT',
    'HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED',
    'HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE',
    'HSA_EXT_IMAGE_CAPABILITY_READ_ONLY',
    'HSA_EXT_IMAGE_CAPABILITY_READ_WRITE',
    'HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_A',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_R', 'HSA_EXT_IMAGE_CHANNEL_ORDER_RA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RG',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8',
    'HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR',
    'HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE', 'HSA_EXT_IMAGE_GEOMETRY_1D',
    'HSA_EXT_IMAGE_GEOMETRY_1DA', 'HSA_EXT_IMAGE_GEOMETRY_1DB',
    'HSA_EXT_IMAGE_GEOMETRY_2D', 'HSA_EXT_IMAGE_GEOMETRY_2DA',
    'HSA_EXT_IMAGE_GEOMETRY_2DADEPTH',
    'HSA_EXT_IMAGE_GEOMETRY_2DDEPTH', 'HSA_EXT_IMAGE_GEOMETRY_3D',
    'HSA_EXT_POINTER_TYPE_GRAPHICS', 'HSA_EXT_POINTER_TYPE_HSA',
    'HSA_EXT_POINTER_TYPE_IPC', 'HSA_EXT_POINTER_TYPE_LOCKED',
    'HSA_EXT_POINTER_TYPE_UNKNOWN',
    'HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_EXT_PROGRAM_INFO_MACHINE_MODEL',
    'HSA_EXT_PROGRAM_INFO_PROFILE',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED',
    'HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED',
    'HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED',
    'HSA_EXT_SAMPLER_FILTER_MODE_LINEAR',
    'HSA_EXT_SAMPLER_FILTER_MODE_NEAREST',
    'HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH',
    'HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED',
    'HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE',
    'HSA_EXT_STATUS_ERROR_INVALID_MODULE',
    'HSA_EXT_STATUS_ERROR_INVALID_PROGRAM',
    'HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED',
    'HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH', 'HSA_FENCE_SCOPE_AGENT',
    'HSA_FENCE_SCOPE_NONE', 'HSA_FENCE_SCOPE_SYSTEM',
    'HSA_FLUSH_MODE_FTZ', 'HSA_FLUSH_MODE_NON_FTZ', 'HSA_FP_TYPE_16',
    'HSA_FP_TYPE_32', 'HSA_FP_TYPE_64', 'HSA_IOMMU_SUPPORT_NONE',
    'HSA_IOMMU_SUPPORT_V2',
    'HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_ISA_INFO_CALL_CONVENTION_COUNT',
    'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT',
    'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE',
    'HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_ISA_INFO_FAST_F16_OPERATION',
    'HSA_ISA_INFO_FBARRIER_MAX_SIZE', 'HSA_ISA_INFO_GRID_MAX_DIM',
    'HSA_ISA_INFO_GRID_MAX_SIZE', 'HSA_ISA_INFO_MACHINE_MODELS',
    'HSA_ISA_INFO_NAME', 'HSA_ISA_INFO_NAME_LENGTH',
    'HSA_ISA_INFO_PROFILES', 'HSA_ISA_INFO_WORKGROUP_MAX_DIM',
    'HSA_ISA_INFO_WORKGROUP_MAX_SIZE',
    'HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS',
    'HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS',
    'HSA_MACHINE_MODEL_LARGE', 'HSA_MACHINE_MODEL_SMALL',
    'HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_BARRIER',
    'HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_TYPE',
    'HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_BARRIER',
    'HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_TYPE', 'HSA_PACKET_TYPE_AGENT_DISPATCH',
    'HSA_PACKET_TYPE_BARRIER_AND', 'HSA_PACKET_TYPE_BARRIER_OR',
    'HSA_PACKET_TYPE_INVALID', 'HSA_PACKET_TYPE_KERNEL_DISPATCH',
    'HSA_PACKET_TYPE_VENDOR_SPECIFIC', 'HSA_PROFILE_BASE',
    'HSA_PROFILE_FULL', 'HSA_QUEUE_FEATURE_AGENT_DISPATCH',
    'HSA_QUEUE_FEATURE_KERNEL_DISPATCH', 'HSA_QUEUE_TYPE_COOPERATIVE',
    'HSA_QUEUE_TYPE_MULTI', 'HSA_QUEUE_TYPE_SINGLE',
    'HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_KERNARG',
    'HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE',
    'HSA_REGION_INFO_ALLOC_MAX_SIZE', 'HSA_REGION_INFO_GLOBAL_FLAGS',
    'HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT',
    'HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED',
    'HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE',
    'HSA_REGION_INFO_SEGMENT', 'HSA_REGION_INFO_SIZE',
    'HSA_REGION_SEGMENT_GLOBAL', 'HSA_REGION_SEGMENT_GROUP',
    'HSA_REGION_SEGMENT_KERNARG', 'HSA_REGION_SEGMENT_PRIVATE',
    'HSA_REGION_SEGMENT_READONLY', 'HSA_ROUND_METHOD_DOUBLE',
    'HSA_ROUND_METHOD_SINGLE', 'HSA_SIGNAL_CONDITION_EQ',
    'HSA_SIGNAL_CONDITION_GTE', 'HSA_SIGNAL_CONDITION_LT',
    'HSA_SIGNAL_CONDITION_NE', 'HSA_STATUS_CU_MASK_REDUCED',
    'HSA_STATUS_ERROR', 'HSA_STATUS_ERROR_EXCEPTION',
    'HSA_STATUS_ERROR_FATAL', 'HSA_STATUS_ERROR_FROZEN_EXECUTABLE',
    'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION',
    'HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS',
    'HSA_STATUS_ERROR_INVALID_AGENT',
    'HSA_STATUS_ERROR_INVALID_ALLOCATION',
    'HSA_STATUS_ERROR_INVALID_ARGUMENT',
    'HSA_STATUS_ERROR_INVALID_CACHE',
    'HSA_STATUS_ERROR_INVALID_CODE_OBJECT',
    'HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER',
    'HSA_STATUS_ERROR_INVALID_CODE_SYMBOL',
    'HSA_STATUS_ERROR_INVALID_EXECUTABLE',
    'HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL',
    'HSA_STATUS_ERROR_INVALID_FILE', 'HSA_STATUS_ERROR_INVALID_INDEX',
    'HSA_STATUS_ERROR_INVALID_ISA',
    'HSA_STATUS_ERROR_INVALID_ISA_NAME',
    'HSA_STATUS_ERROR_INVALID_MEMORY_POOL',
    'HSA_STATUS_ERROR_INVALID_PACKET_FORMAT',
    'HSA_STATUS_ERROR_INVALID_QUEUE',
    'HSA_STATUS_ERROR_INVALID_QUEUE_CREATION',
    'HSA_STATUS_ERROR_INVALID_REGION',
    'HSA_STATUS_ERROR_INVALID_RUNTIME_STATE',
    'HSA_STATUS_ERROR_INVALID_SIGNAL',
    'HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP',
    'HSA_STATUS_ERROR_INVALID_SYMBOL_NAME',
    'HSA_STATUS_ERROR_INVALID_WAVEFRONT',
    'HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION',
    'HSA_STATUS_ERROR_MEMORY_FAULT',
    'HSA_STATUS_ERROR_NOT_INITIALIZED',
    'HSA_STATUS_ERROR_OUT_OF_REGISTERS',
    'HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    'HSA_STATUS_ERROR_REFCOUNT_OVERFLOW',
    'HSA_STATUS_ERROR_RESOURCE_FREE',
    'HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED',
    'HSA_STATUS_ERROR_VARIABLE_UNDEFINED', 'HSA_STATUS_INFO_BREAK',
    'HSA_STATUS_SUCCESS', 'HSA_SYMBOL_KIND_INDIRECT_FUNCTION',
    'HSA_SYMBOL_KIND_KERNEL', 'HSA_SYMBOL_KIND_VARIABLE',
    'HSA_SYMBOL_LINKAGE_MODULE', 'HSA_SYMBOL_LINKAGE_PROGRAM',
    'HSA_SYSTEM_INFO_ENDIANNESS', 'HSA_SYSTEM_INFO_EXTENSIONS',
    'HSA_SYSTEM_INFO_MACHINE_MODEL',
    'HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT', 'HSA_SYSTEM_INFO_TIMESTAMP',
    'HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY',
    'HSA_SYSTEM_INFO_VERSION_MAJOR', 'HSA_SYSTEM_INFO_VERSION_MINOR',
    'HSA_VARIABLE_ALLOCATION_AGENT',
    'HSA_VARIABLE_ALLOCATION_PROGRAM', 'HSA_VARIABLE_SEGMENT_GLOBAL',
    'HSA_VARIABLE_SEGMENT_READONLY', 'HSA_WAIT_STATE_ACTIVE',
    'HSA_WAIT_STATE_BLOCKED', 'HSA_WAVEFRONT_INFO_SIZE',
    'MEMORY_TYPE_NONE', 'MEMORY_TYPE_PINNED',
    'enum_hsa_ext_amd_h_179', 'enum_hsa_ext_finalize_h_69',
    'enum_hsa_ext_image_h_68', 'enum_hsa_ext_image_h_93',
    'hsaDeviceToDevice', 'hsaDeviceToHost', 'hsaHostToDevice',
    'hsaHostToHost', 'hsa_access_permission_t',
    'hsa_agent_dispatch_packet_t', 'hsa_agent_extension_supported',
    'hsa_agent_feature_t', 'hsa_agent_get_exception_policies',
    'hsa_agent_get_info', 'hsa_agent_info_t',
    'hsa_agent_iterate_caches', 'hsa_agent_iterate_isas',
    'hsa_agent_iterate_regions',
    'hsa_agent_major_extension_supported', 'hsa_agent_t',
    'hsa_amd_agent_info_s', 'hsa_amd_agent_info_t',
    'hsa_amd_agent_info_t__enumvalues',
    'hsa_amd_agent_iterate_memory_pools',
    'hsa_amd_agent_memory_pool_get_info',
    'hsa_amd_agent_memory_pool_info_t', 'hsa_amd_agents_allow_access',
    'hsa_amd_async_function', 'hsa_amd_barrier_value_packet_t',
    'hsa_amd_coherency_get_type', 'hsa_amd_coherency_set_type',
    'hsa_amd_coherency_type_s', 'hsa_amd_coherency_type_t',
    'hsa_amd_coherency_type_t__enumvalues',
    'hsa_amd_copy_direction_t', 'hsa_amd_deallocation_callback_t',
    'hsa_amd_deregister_deallocation_callback', 'hsa_amd_event_t',
    'hsa_amd_event_type_s', 'hsa_amd_event_type_t',
    'hsa_amd_event_type_t__enumvalues',
    'hsa_amd_gpu_hw_exception_info_t',
    'hsa_amd_gpu_memory_fault_info_t', 'hsa_amd_hdp_flush_t',
    'hsa_amd_hw_exception_reset_cause_t',
    'hsa_amd_hw_exception_reset_type_t', 'hsa_amd_image_create',
    'hsa_amd_image_descriptor_t', 'hsa_amd_image_get_info_max_dim',
    'hsa_amd_interop_map_buffer', 'hsa_amd_interop_unmap_buffer',
    'hsa_amd_iommu_version_t', 'hsa_amd_ipc_memory_attach',
    'hsa_amd_ipc_memory_create', 'hsa_amd_ipc_memory_detach',
    'hsa_amd_ipc_memory_t', 'hsa_amd_ipc_signal_attach',
    'hsa_amd_ipc_signal_create', 'hsa_amd_ipc_signal_t',
    'hsa_amd_link_info_type_t', 'hsa_amd_memory_access_desc_t',
    'hsa_amd_memory_async_copy',
    'hsa_amd_memory_async_copy_on_engine',
    'hsa_amd_memory_async_copy_rect',
    'hsa_amd_memory_copy_engine_status',
    'hsa_amd_memory_fault_reason_t', 'hsa_amd_memory_fill',
    'hsa_amd_memory_lock', 'hsa_amd_memory_lock_to_pool',
    'hsa_amd_memory_migrate', 'hsa_amd_memory_pool_access_t',
    'hsa_amd_memory_pool_allocate', 'hsa_amd_memory_pool_can_migrate',
    'hsa_amd_memory_pool_flag_s', 'hsa_amd_memory_pool_flag_t',
    'hsa_amd_memory_pool_flag_t__enumvalues',
    'hsa_amd_memory_pool_free', 'hsa_amd_memory_pool_get_info',
    'hsa_amd_memory_pool_global_flag_s',
    'hsa_amd_memory_pool_global_flag_t',
    'hsa_amd_memory_pool_global_flag_t__enumvalues',
    'hsa_amd_memory_pool_info_t', 'hsa_amd_memory_pool_link_info_t',
    'hsa_amd_memory_pool_location_s',
    'hsa_amd_memory_pool_location_t',
    'hsa_amd_memory_pool_location_t__enumvalues',
    'hsa_amd_memory_pool_t', 'hsa_amd_memory_type_t',
    'hsa_amd_memory_unlock', 'hsa_amd_packet_type8_t',
    'hsa_amd_packet_type_t', 'hsa_amd_pointer_info',
    'hsa_amd_pointer_info_set_userdata', 'hsa_amd_pointer_info_t',
    'hsa_amd_pointer_type_t', 'hsa_amd_portable_close_dmabuf',
    'hsa_amd_portable_export_dmabuf',
    'hsa_amd_profiling_async_copy_enable',
    'hsa_amd_profiling_async_copy_time_t',
    'hsa_amd_profiling_convert_tick_to_system_domain',
    'hsa_amd_profiling_dispatch_time_t',
    'hsa_amd_profiling_get_async_copy_time',
    'hsa_amd_profiling_get_dispatch_time',
    'hsa_amd_profiling_set_profiler_enabled',
    'hsa_amd_queue_cu_get_mask', 'hsa_amd_queue_cu_set_mask',
    'hsa_amd_queue_priority_s', 'hsa_amd_queue_priority_t',
    'hsa_amd_queue_priority_t__enumvalues',
    'hsa_amd_queue_set_priority', 'hsa_amd_region_info_s',
    'hsa_amd_region_info_t', 'hsa_amd_region_info_t__enumvalues',
    'hsa_amd_register_deallocation_callback',
    'hsa_amd_register_system_event_handler', 'hsa_amd_sdma_engine_id',
    'hsa_amd_sdma_engine_id_t',
    'hsa_amd_sdma_engine_id_t__enumvalues', 'hsa_amd_segment_t',
    'hsa_amd_signal_async_handler', 'hsa_amd_signal_attribute_t',
    'hsa_amd_signal_create', 'hsa_amd_signal_handler',
    'hsa_amd_signal_value_pointer', 'hsa_amd_signal_wait_any',
    'hsa_amd_spm_acquire', 'hsa_amd_spm_release',
    'hsa_amd_spm_set_dest_buffer', 'hsa_amd_svm_attribute_pair_t',
    'hsa_amd_svm_attribute_s', 'hsa_amd_svm_attribute_t',
    'hsa_amd_svm_attribute_t__enumvalues',
    'hsa_amd_svm_attributes_get', 'hsa_amd_svm_attributes_set',
    'hsa_amd_svm_model_s', 'hsa_amd_svm_model_t',
    'hsa_amd_svm_model_t__enumvalues', 'hsa_amd_svm_prefetch_async',
    'hsa_amd_system_event_callback_t',
    'hsa_amd_vendor_packet_header_t', 'hsa_amd_vmem_address_free',
    'hsa_amd_vmem_address_reserve', 'hsa_amd_vmem_alloc_handle_t',
    'hsa_amd_vmem_export_shareable_handle', 'hsa_amd_vmem_get_access',
    'hsa_amd_vmem_get_alloc_properties_from_handle',
    'hsa_amd_vmem_handle_create', 'hsa_amd_vmem_handle_release',
    'hsa_amd_vmem_import_shareable_handle', 'hsa_amd_vmem_map',
    'hsa_amd_vmem_retain_alloc_handle', 'hsa_amd_vmem_set_access',
    'hsa_amd_vmem_unmap', 'hsa_barrier_and_packet_t',
    'hsa_barrier_or_packet_t', 'hsa_cache_get_info',
    'hsa_cache_info_t', 'hsa_cache_t', 'hsa_callback_data_t',
    'hsa_code_object_deserialize', 'hsa_code_object_destroy',
    'hsa_code_object_get_info', 'hsa_code_object_get_symbol',
    'hsa_code_object_get_symbol_from_name', 'hsa_code_object_info_t',
    'hsa_code_object_iterate_symbols',
    'hsa_code_object_reader_create_from_file',
    'hsa_code_object_reader_create_from_memory',
    'hsa_code_object_reader_destroy', 'hsa_code_object_reader_t',
    'hsa_code_object_serialize', 'hsa_code_object_t',
    'hsa_code_object_type_t', 'hsa_code_symbol_get_info',
    'hsa_code_symbol_info_t', 'hsa_code_symbol_t',
    'hsa_default_float_rounding_mode_t', 'hsa_device_type_t',
    'hsa_dim3_t', 'hsa_endianness_t', 'hsa_exception_policy_t',
    'hsa_executable_agent_global_variable_define',
    'hsa_executable_create', 'hsa_executable_create_alt',
    'hsa_executable_destroy', 'hsa_executable_freeze',
    'hsa_executable_get_info', 'hsa_executable_get_symbol',
    'hsa_executable_get_symbol_by_name',
    'hsa_executable_global_variable_define', 'hsa_executable_info_t',
    'hsa_executable_iterate_agent_symbols',
    'hsa_executable_iterate_program_symbols',
    'hsa_executable_iterate_symbols',
    'hsa_executable_load_agent_code_object',
    'hsa_executable_load_code_object',
    'hsa_executable_load_program_code_object',
    'hsa_executable_readonly_variable_define',
    'hsa_executable_state_t', 'hsa_executable_symbol_get_info',
    'hsa_executable_symbol_info_t', 'hsa_executable_symbol_t',
    'hsa_executable_t', 'hsa_executable_validate',
    'hsa_executable_validate_alt', 'hsa_ext_control_directives_t',
    'hsa_ext_finalizer_1_00_pfn_t',
    'hsa_ext_finalizer_call_convention_t',
    'hsa_ext_image_capability_t', 'hsa_ext_image_channel_order32_t',
    'hsa_ext_image_channel_order_t', 'hsa_ext_image_channel_type32_t',
    'hsa_ext_image_channel_type_t', 'hsa_ext_image_clear',
    'hsa_ext_image_copy', 'hsa_ext_image_create',
    'hsa_ext_image_create_with_layout', 'hsa_ext_image_data_get_info',
    'hsa_ext_image_data_get_info_with_layout',
    'hsa_ext_image_data_info_t', 'hsa_ext_image_data_layout_t',
    'hsa_ext_image_descriptor_t', 'hsa_ext_image_destroy',
    'hsa_ext_image_export', 'hsa_ext_image_format_t',
    'hsa_ext_image_geometry_t', 'hsa_ext_image_get_capability',
    'hsa_ext_image_get_capability_with_layout',
    'hsa_ext_image_import', 'hsa_ext_image_region_t',
    'hsa_ext_image_t', 'hsa_ext_images_1_00_pfn_t',
    'hsa_ext_images_1_pfn_t', 'hsa_ext_module_t',
    'hsa_ext_program_add_module', 'hsa_ext_program_create',
    'hsa_ext_program_destroy', 'hsa_ext_program_finalize',
    'hsa_ext_program_get_info', 'hsa_ext_program_info_t',
    'hsa_ext_program_iterate_modules', 'hsa_ext_program_t',
    'hsa_ext_sampler_addressing_mode32_t',
    'hsa_ext_sampler_addressing_mode_t',
    'hsa_ext_sampler_coordinate_mode32_t',
    'hsa_ext_sampler_coordinate_mode_t', 'hsa_ext_sampler_create',
    'hsa_ext_sampler_descriptor_t', 'hsa_ext_sampler_destroy',
    'hsa_ext_sampler_filter_mode32_t',
    'hsa_ext_sampler_filter_mode_t', 'hsa_ext_sampler_t',
    'hsa_extension_get_name', 'hsa_extension_t', 'hsa_fence_scope_t',
    'hsa_file_t', 'hsa_flush_mode_t', 'hsa_fp_type_t', 'hsa_init',
    'hsa_isa_compatible', 'hsa_isa_from_name',
    'hsa_isa_get_exception_policies', 'hsa_isa_get_info',
    'hsa_isa_get_info_alt', 'hsa_isa_get_round_method',
    'hsa_isa_info_t', 'hsa_isa_iterate_wavefronts', 'hsa_isa_t',
    'hsa_iterate_agents', 'hsa_kernel_dispatch_packet_setup_t',
    'hsa_kernel_dispatch_packet_setup_width_t',
    'hsa_kernel_dispatch_packet_t', 'hsa_loaded_code_object_t',
    'hsa_machine_model_t', 'hsa_memory_allocate',
    'hsa_memory_assign_agent', 'hsa_memory_copy',
    'hsa_memory_deregister', 'hsa_memory_free', 'hsa_memory_register',
    'hsa_packet_header_t', 'hsa_packet_header_width_t',
    'hsa_packet_type_t', 'hsa_pitched_ptr_t', 'hsa_profile_t',
    'hsa_queue_add_write_index_acq_rel',
    'hsa_queue_add_write_index_acquire',
    'hsa_queue_add_write_index_relaxed',
    'hsa_queue_add_write_index_release',
    'hsa_queue_add_write_index_scacq_screl',
    'hsa_queue_add_write_index_scacquire',
    'hsa_queue_add_write_index_screlease',
    'hsa_queue_cas_write_index_acq_rel',
    'hsa_queue_cas_write_index_acquire',
    'hsa_queue_cas_write_index_relaxed',
    'hsa_queue_cas_write_index_release',
    'hsa_queue_cas_write_index_scacq_screl',
    'hsa_queue_cas_write_index_scacquire',
    'hsa_queue_cas_write_index_screlease', 'hsa_queue_create',
    'hsa_queue_destroy', 'hsa_queue_feature_t',
    'hsa_queue_inactivate', 'hsa_queue_load_read_index_acquire',
    'hsa_queue_load_read_index_relaxed',
    'hsa_queue_load_read_index_scacquire',
    'hsa_queue_load_write_index_acquire',
    'hsa_queue_load_write_index_relaxed',
    'hsa_queue_load_write_index_scacquire',
    'hsa_queue_store_read_index_relaxed',
    'hsa_queue_store_read_index_release',
    'hsa_queue_store_read_index_screlease',
    'hsa_queue_store_write_index_relaxed',
    'hsa_queue_store_write_index_release',
    'hsa_queue_store_write_index_screlease', 'hsa_queue_t',
    'hsa_queue_type32_t', 'hsa_queue_type_t', 'hsa_region_get_info',
    'hsa_region_global_flag_t', 'hsa_region_info_t',
    'hsa_region_segment_t', 'hsa_region_t', 'hsa_round_method_t',
    'hsa_shut_down', 'hsa_signal_add_acq_rel',
    'hsa_signal_add_acquire', 'hsa_signal_add_relaxed',
    'hsa_signal_add_release', 'hsa_signal_add_scacq_screl',
    'hsa_signal_add_scacquire', 'hsa_signal_add_screlease',
    'hsa_signal_and_acq_rel', 'hsa_signal_and_acquire',
    'hsa_signal_and_relaxed', 'hsa_signal_and_release',
    'hsa_signal_and_scacq_screl', 'hsa_signal_and_scacquire',
    'hsa_signal_and_screlease', 'hsa_signal_cas_acq_rel',
    'hsa_signal_cas_acquire', 'hsa_signal_cas_relaxed',
    'hsa_signal_cas_release', 'hsa_signal_cas_scacq_screl',
    'hsa_signal_cas_scacquire', 'hsa_signal_cas_screlease',
    'hsa_signal_condition32_t', 'hsa_signal_condition_t',
    'hsa_signal_create', 'hsa_signal_destroy',
    'hsa_signal_exchange_acq_rel', 'hsa_signal_exchange_acquire',
    'hsa_signal_exchange_relaxed', 'hsa_signal_exchange_release',
    'hsa_signal_exchange_scacq_screl',
    'hsa_signal_exchange_scacquire', 'hsa_signal_exchange_screlease',
    'hsa_signal_group_create', 'hsa_signal_group_destroy',
    'hsa_signal_group_t', 'hsa_signal_group_wait_any_relaxed',
    'hsa_signal_group_wait_any_scacquire', 'hsa_signal_load_acquire',
    'hsa_signal_load_relaxed', 'hsa_signal_load_scacquire',
    'hsa_signal_or_acq_rel', 'hsa_signal_or_acquire',
    'hsa_signal_or_relaxed', 'hsa_signal_or_release',
    'hsa_signal_or_scacq_screl', 'hsa_signal_or_scacquire',
    'hsa_signal_or_screlease', 'hsa_signal_silent_store_relaxed',
    'hsa_signal_silent_store_screlease', 'hsa_signal_store_relaxed',
    'hsa_signal_store_release', 'hsa_signal_store_screlease',
    'hsa_signal_subtract_acq_rel', 'hsa_signal_subtract_acquire',
    'hsa_signal_subtract_relaxed', 'hsa_signal_subtract_release',
    'hsa_signal_subtract_scacq_screl',
    'hsa_signal_subtract_scacquire', 'hsa_signal_subtract_screlease',
    'hsa_signal_t', 'hsa_signal_value_t', 'hsa_signal_wait_acquire',
    'hsa_signal_wait_relaxed', 'hsa_signal_wait_scacquire',
    'hsa_signal_xor_acq_rel', 'hsa_signal_xor_acquire',
    'hsa_signal_xor_relaxed', 'hsa_signal_xor_release',
    'hsa_signal_xor_scacq_screl', 'hsa_signal_xor_scacquire',
    'hsa_signal_xor_screlease', 'hsa_soft_queue_create',
    'hsa_status_string', 'hsa_status_t', 'hsa_symbol_kind_t',
    'hsa_symbol_linkage_t', 'hsa_system_extension_supported',
    'hsa_system_get_extension_table', 'hsa_system_get_info',
    'hsa_system_get_major_extension_table', 'hsa_system_info_t',
    'hsa_system_major_extension_supported',
    'hsa_variable_allocation_t', 'hsa_variable_segment_t',
    'hsa_wait_state_t', 'hsa_wavefront_get_info',
    'hsa_wavefront_info_t', 'hsa_wavefront_t', 'int32_t', 'size_t',
    'struct_BrigModuleHeader', 'struct_hsa_agent_dispatch_packet_s',
    'struct_hsa_agent_s', 'struct_hsa_amd_barrier_value_packet_s',
    'struct_hsa_amd_event_s',
    'struct_hsa_amd_gpu_hw_exception_info_s',
    'struct_hsa_amd_gpu_memory_fault_info_s',
    'struct_hsa_amd_hdp_flush_s', 'struct_hsa_amd_image_descriptor_s',
    'struct_hsa_amd_ipc_memory_s',
    'struct_hsa_amd_memory_access_desc_s',
    'struct_hsa_amd_memory_pool_link_info_s',
    'struct_hsa_amd_memory_pool_s', 'struct_hsa_amd_packet_header_s',
    'struct_hsa_amd_pointer_info_s',
    'struct_hsa_amd_profiling_async_copy_time_s',
    'struct_hsa_amd_profiling_dispatch_time_s',
    'struct_hsa_amd_svm_attribute_pair_s',
    'struct_hsa_amd_vmem_alloc_handle_s',
    'struct_hsa_barrier_and_packet_s',
    'struct_hsa_barrier_or_packet_s', 'struct_hsa_cache_s',
    'struct_hsa_callback_data_s', 'struct_hsa_code_object_reader_s',
    'struct_hsa_code_object_s', 'struct_hsa_code_symbol_s',
    'struct_hsa_dim3_s', 'struct_hsa_executable_s',
    'struct_hsa_executable_symbol_s',
    'struct_hsa_ext_control_directives_s',
    'struct_hsa_ext_finalizer_1_00_pfn_s',
    'struct_hsa_ext_image_data_info_s',
    'struct_hsa_ext_image_descriptor_s',
    'struct_hsa_ext_image_format_s', 'struct_hsa_ext_image_region_s',
    'struct_hsa_ext_image_s', 'struct_hsa_ext_images_1_00_pfn_s',
    'struct_hsa_ext_images_1_pfn_s', 'struct_hsa_ext_program_s',
    'struct_hsa_ext_sampler_descriptor_s', 'struct_hsa_ext_sampler_s',
    'struct_hsa_isa_s', 'struct_hsa_kernel_dispatch_packet_s',
    'struct_hsa_loaded_code_object_s', 'struct_hsa_pitched_ptr_s',
    'struct_hsa_queue_s', 'struct_hsa_region_s',
    'struct_hsa_signal_group_s', 'struct_hsa_signal_s',
    'struct_hsa_wavefront_s', 'uint16_t', 'uint32_t', 'uint64_t',
    'union_union_hsa_ext_amd_h_2329']


# tinygrad/runtime/autogen/opencl.py

# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



_libraries = {}
_libraries['libOpenCL.so.1'] = ctypes.CDLL(ctypes.util.find_library('OpenCL'))
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





__OPENCL_CL_H = True # macro
CL_NAME_VERSION_MAX_NAME_SIZE = 64 # macro
CL_SUCCESS = 0 # macro
CL_DEVICE_NOT_FOUND = -1 # macro
CL_DEVICE_NOT_AVAILABLE = -2 # macro
CL_COMPILER_NOT_AVAILABLE = -3 # macro
CL_MEM_OBJECT_ALLOCATION_FAILURE = -4 # macro
CL_OUT_OF_RESOURCES = -5 # macro
CL_OUT_OF_HOST_MEMORY = -6 # macro
CL_PROFILING_INFO_NOT_AVAILABLE = -7 # macro
CL_MEM_COPY_OVERLAP = -8 # macro
CL_IMAGE_FORMAT_MISMATCH = -9 # macro
CL_IMAGE_FORMAT_NOT_SUPPORTED = -10 # macro
CL_BUILD_PROGRAM_FAILURE = -11 # macro
CL_MAP_FAILURE = -12 # macro
CL_MISALIGNED_SUB_BUFFER_OFFSET = -13 # macro
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14 # macro
CL_COMPILE_PROGRAM_FAILURE = -15 # macro
CL_LINKER_NOT_AVAILABLE = -16 # macro
CL_LINK_PROGRAM_FAILURE = -17 # macro
CL_DEVICE_PARTITION_FAILED = -18 # macro
CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19 # macro
CL_INVALID_VALUE = -30 # macro
CL_INVALID_DEVICE_TYPE = -31 # macro
CL_INVALID_PLATFORM = -32 # macro
CL_INVALID_DEVICE = -33 # macro
CL_INVALID_CONTEXT = -34 # macro
CL_INVALID_QUEUE_PROPERTIES = -35 # macro
CL_INVALID_COMMAND_QUEUE = -36 # macro
CL_INVALID_HOST_PTR = -37 # macro
CL_INVALID_MEM_OBJECT = -38 # macro
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39 # macro
CL_INVALID_IMAGE_SIZE = -40 # macro
CL_INVALID_SAMPLER = -41 # macro
CL_INVALID_BINARY = -42 # macro
CL_INVALID_BUILD_OPTIONS = -43 # macro
CL_INVALID_PROGRAM = -44 # macro
CL_INVALID_PROGRAM_EXECUTABLE = -45 # macro
CL_INVALID_KERNEL_NAME = -46 # macro
CL_INVALID_KERNEL_DEFINITION = -47 # macro
CL_INVALID_KERNEL = -48 # macro
CL_INVALID_ARG_INDEX = -49 # macro
CL_INVALID_ARG_VALUE = -50 # macro
CL_INVALID_ARG_SIZE = -51 # macro
CL_INVALID_KERNEL_ARGS = -52 # macro
CL_INVALID_WORK_DIMENSION = -53 # macro
CL_INVALID_WORK_GROUP_SIZE = -54 # macro
CL_INVALID_WORK_ITEM_SIZE = -55 # macro
CL_INVALID_GLOBAL_OFFSET = -56 # macro
CL_INVALID_EVENT_WAIT_LIST = -57 # macro
CL_INVALID_EVENT = -58 # macro
CL_INVALID_OPERATION = -59 # macro
CL_INVALID_GL_OBJECT = -60 # macro
CL_INVALID_BUFFER_SIZE = -61 # macro
CL_INVALID_MIP_LEVEL = -62 # macro
CL_INVALID_GLOBAL_WORK_SIZE = -63 # macro
CL_INVALID_PROPERTY = -64 # macro
CL_INVALID_IMAGE_DESCRIPTOR = -65 # macro
CL_INVALID_COMPILER_OPTIONS = -66 # macro
CL_INVALID_LINKER_OPTIONS = -67 # macro
CL_INVALID_DEVICE_PARTITION_COUNT = -68 # macro
CL_INVALID_PIPE_SIZE = -69 # macro
CL_INVALID_DEVICE_QUEUE = -70 # macro
CL_INVALID_SPEC_ID = -71 # macro
CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72 # macro
CL_FALSE = 0 # macro
CL_TRUE = 1 # macro
CL_BLOCKING = 1 # macro
CL_NON_BLOCKING = 0 # macro
CL_PLATFORM_PROFILE = 0x0900 # macro
CL_PLATFORM_VERSION = 0x0901 # macro
CL_PLATFORM_NAME = 0x0902 # macro
CL_PLATFORM_VENDOR = 0x0903 # macro
CL_PLATFORM_EXTENSIONS = 0x0904 # macro
CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905 # macro
CL_PLATFORM_NUMERIC_VERSION = 0x0906 # macro
CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907 # macro
CL_DEVICE_TYPE_DEFAULT = (1<<0) # macro
CL_DEVICE_TYPE_CPU = (1<<1) # macro
CL_DEVICE_TYPE_GPU = (1<<2) # macro
CL_DEVICE_TYPE_ACCELERATOR = (1<<3) # macro
CL_DEVICE_TYPE_CUSTOM = (1<<4) # macro
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF # macro
CL_DEVICE_TYPE = 0x1000 # macro
CL_DEVICE_VENDOR_ID = 0x1001 # macro
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002 # macro
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003 # macro
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004 # macro
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B # macro
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C # macro
CL_DEVICE_ADDRESS_BITS = 0x100D # macro
CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E # macro
CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F # macro
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010 # macro
CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011 # macro
CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012 # macro
CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013 # macro
CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014 # macro
CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015 # macro
CL_DEVICE_IMAGE_SUPPORT = 0x1016 # macro
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017 # macro
CL_DEVICE_MAX_SAMPLERS = 0x1018 # macro
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019 # macro
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A # macro
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B # macro
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C # macro
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D # macro
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E # macro
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F # macro
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020 # macro
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021 # macro
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022 # macro
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023 # macro
CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024 # macro
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025 # macro
CL_DEVICE_ENDIAN_LITTLE = 0x1026 # macro
CL_DEVICE_AVAILABLE = 0x1027 # macro
CL_DEVICE_COMPILER_AVAILABLE = 0x1028 # macro
CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029 # macro
CL_DEVICE_QUEUE_PROPERTIES = 0x102A # macro
CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A # macro
CL_DEVICE_NAME = 0x102B # macro
CL_DEVICE_VENDOR = 0x102C # macro
CL_DRIVER_VERSION = 0x102D # macro
CL_DEVICE_PROFILE = 0x102E # macro
CL_DEVICE_VERSION = 0x102F # macro
CL_DEVICE_EXTENSIONS = 0x1030 # macro
CL_DEVICE_PLATFORM = 0x1031 # macro
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032 # macro
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034 # macro
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039 # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B # macro
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C # macro
CL_DEVICE_OPENCL_C_VERSION = 0x103D # macro
CL_DEVICE_LINKER_AVAILABLE = 0x103E # macro
CL_DEVICE_BUILT_IN_KERNELS = 0x103F # macro
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040 # macro
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041 # macro
CL_DEVICE_PARENT_DEVICE = 0x1042 # macro
CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043 # macro
CL_DEVICE_PARTITION_PROPERTIES = 0x1044 # macro
CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045 # macro
CL_DEVICE_PARTITION_TYPE = 0x1046 # macro
CL_DEVICE_REFERENCE_COUNT = 0x1047 # macro
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048 # macro
CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049 # macro
CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A # macro
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B # macro
CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C # macro
CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D # macro
CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E # macro
CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F # macro
CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050 # macro
CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051 # macro
CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052 # macro
CL_DEVICE_SVM_CAPABILITIES = 0x1053 # macro
CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054 # macro
CL_DEVICE_MAX_PIPE_ARGS = 0x1055 # macro
CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056 # macro
CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057 # macro
CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058 # macro
CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059 # macro
CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A # macro
CL_DEVICE_IL_VERSION = 0x105B # macro
CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C # macro
CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D # macro
CL_DEVICE_NUMERIC_VERSION = 0x105E # macro
CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060 # macro
CL_DEVICE_ILS_WITH_VERSION = 0x1061 # macro
CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062 # macro
CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063 # macro
CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064 # macro
CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065 # macro
CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066 # macro
CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067 # macro
CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068 # macro
CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069 # macro
CL_DEVICE_OPENCL_C_FEATURES = 0x106F # macro
CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070 # macro
CL_DEVICE_PIPE_SUPPORT = 0x1071 # macro
CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072 # macro
CL_FP_DENORM = (1<<0) # macro
CL_FP_INF_NAN = (1<<1) # macro
CL_FP_ROUND_TO_NEAREST = (1<<2) # macro
CL_FP_ROUND_TO_ZERO = (1<<3) # macro
CL_FP_ROUND_TO_INF = (1<<4) # macro
CL_FP_FMA = (1<<5) # macro
CL_FP_SOFT_FLOAT = (1<<6) # macro
CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = (1<<7) # macro
CL_NONE = 0x0 # macro
CL_READ_ONLY_CACHE = 0x1 # macro
CL_READ_WRITE_CACHE = 0x2 # macro
CL_LOCAL = 0x1 # macro
CL_GLOBAL = 0x2 # macro
CL_EXEC_KERNEL = (1<<0) # macro
CL_EXEC_NATIVE_KERNEL = (1<<1) # macro
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1<<0) # macro
CL_QUEUE_PROFILING_ENABLE = (1<<1) # macro
CL_QUEUE_ON_DEVICE = (1<<2) # macro
CL_QUEUE_ON_DEVICE_DEFAULT = (1<<3) # macro
CL_CONTEXT_REFERENCE_COUNT = 0x1080 # macro
CL_CONTEXT_DEVICES = 0x1081 # macro
CL_CONTEXT_PROPERTIES = 0x1082 # macro
CL_CONTEXT_NUM_DEVICES = 0x1083 # macro
CL_CONTEXT_PLATFORM = 0x1084 # macro
CL_CONTEXT_INTEROP_USER_SYNC = 0x1085 # macro
CL_DEVICE_PARTITION_EQUALLY = 0x1086 # macro
CL_DEVICE_PARTITION_BY_COUNTS = 0x1087 # macro
CL_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x0 # macro
CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088 # macro
CL_DEVICE_AFFINITY_DOMAIN_NUMA = (1<<0) # macro
CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE = (1<<1) # macro
CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE = (1<<2) # macro
CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE = (1<<3) # macro
CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE = (1<<4) # macro
CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1<<5) # macro
CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = (1<<0) # macro
CL_DEVICE_SVM_FINE_GRAIN_BUFFER = (1<<1) # macro
CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = (1<<2) # macro
CL_DEVICE_SVM_ATOMICS = (1<<3) # macro
CL_QUEUE_CONTEXT = 0x1090 # macro
CL_QUEUE_DEVICE = 0x1091 # macro
CL_QUEUE_REFERENCE_COUNT = 0x1092 # macro
CL_QUEUE_PROPERTIES = 0x1093 # macro
CL_QUEUE_SIZE = 0x1094 # macro
CL_QUEUE_DEVICE_DEFAULT = 0x1095 # macro
CL_QUEUE_PROPERTIES_ARRAY = 0x1098 # macro
CL_MEM_READ_WRITE = (1<<0) # macro
CL_MEM_WRITE_ONLY = (1<<1) # macro
CL_MEM_READ_ONLY = (1<<2) # macro
CL_MEM_USE_HOST_PTR = (1<<3) # macro
CL_MEM_ALLOC_HOST_PTR = (1<<4) # macro
CL_MEM_COPY_HOST_PTR = (1<<5) # macro
CL_MEM_HOST_WRITE_ONLY = (1<<7) # macro
CL_MEM_HOST_READ_ONLY = (1<<8) # macro
CL_MEM_HOST_NO_ACCESS = (1<<9) # macro
CL_MEM_SVM_FINE_GRAIN_BUFFER = (1<<10) # macro
CL_MEM_SVM_ATOMICS = (1<<11) # macro
CL_MEM_KERNEL_READ_AND_WRITE = (1<<12) # macro
CL_MIGRATE_MEM_OBJECT_HOST = (1<<0) # macro
CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = (1<<1) # macro
CL_R = 0x10B0 # macro
CL_A = 0x10B1 # macro
CL_RG = 0x10B2 # macro
CL_RA = 0x10B3 # macro
CL_RGB = 0x10B4 # macro
CL_RGBA = 0x10B5 # macro
CL_BGRA = 0x10B6 # macro
CL_ARGB = 0x10B7 # macro
CL_INTENSITY = 0x10B8 # macro
CL_LUMINANCE = 0x10B9 # macro
CL_Rx = 0x10BA # macro
CL_RGx = 0x10BB # macro
CL_RGBx = 0x10BC # macro
CL_DEPTH = 0x10BD # macro
CL_DEPTH_STENCIL = 0x10BE # macro
CL_sRGB = 0x10BF # macro
CL_sRGBx = 0x10C0 # macro
CL_sRGBA = 0x10C1 # macro
CL_sBGRA = 0x10C2 # macro
CL_ABGR = 0x10C3 # macro
CL_SNORM_INT8 = 0x10D0 # macro
CL_SNORM_INT16 = 0x10D1 # macro
CL_UNORM_INT8 = 0x10D2 # macro
CL_UNORM_INT16 = 0x10D3 # macro
CL_UNORM_SHORT_565 = 0x10D4 # macro
CL_UNORM_SHORT_555 = 0x10D5 # macro
CL_UNORM_INT_101010 = 0x10D6 # macro
CL_SIGNED_INT8 = 0x10D7 # macro
CL_SIGNED_INT16 = 0x10D8 # macro
CL_SIGNED_INT32 = 0x10D9 # macro
CL_UNSIGNED_INT8 = 0x10DA # macro
CL_UNSIGNED_INT16 = 0x10DB # macro
CL_UNSIGNED_INT32 = 0x10DC # macro
CL_HALF_FLOAT = 0x10DD # macro
CL_FLOAT = 0x10DE # macro
CL_UNORM_INT24 = 0x10DF # macro
CL_UNORM_INT_101010_2 = 0x10E0 # macro
CL_MEM_OBJECT_BUFFER = 0x10F0 # macro
CL_MEM_OBJECT_IMAGE2D = 0x10F1 # macro
CL_MEM_OBJECT_IMAGE3D = 0x10F2 # macro
CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10F3 # macro
CL_MEM_OBJECT_IMAGE1D = 0x10F4 # macro
CL_MEM_OBJECT_IMAGE1D_ARRAY = 0x10F5 # macro
CL_MEM_OBJECT_IMAGE1D_BUFFER = 0x10F6 # macro
CL_MEM_OBJECT_PIPE = 0x10F7 # macro
CL_MEM_TYPE = 0x1100 # macro
CL_MEM_FLAGS = 0x1101 # macro
CL_MEM_SIZE = 0x1102 # macro
CL_MEM_HOST_PTR = 0x1103 # macro
CL_MEM_MAP_COUNT = 0x1104 # macro
CL_MEM_REFERENCE_COUNT = 0x1105 # macro
CL_MEM_CONTEXT = 0x1106 # macro
CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107 # macro
CL_MEM_OFFSET = 0x1108 # macro
CL_MEM_USES_SVM_POINTER = 0x1109 # macro
CL_MEM_PROPERTIES = 0x110A # macro
CL_IMAGE_FORMAT = 0x1110 # macro
CL_IMAGE_ELEMENT_SIZE = 0x1111 # macro
CL_IMAGE_ROW_PITCH = 0x1112 # macro
CL_IMAGE_SLICE_PITCH = 0x1113 # macro
CL_IMAGE_WIDTH = 0x1114 # macro
CL_IMAGE_HEIGHT = 0x1115 # macro
CL_IMAGE_DEPTH = 0x1116 # macro
CL_IMAGE_ARRAY_SIZE = 0x1117 # macro
CL_IMAGE_BUFFER = 0x1118 # macro
CL_IMAGE_NUM_MIP_LEVELS = 0x1119 # macro
CL_IMAGE_NUM_SAMPLES = 0x111A # macro
CL_PIPE_PACKET_SIZE = 0x1120 # macro
CL_PIPE_MAX_PACKETS = 0x1121 # macro
CL_PIPE_PROPERTIES = 0x1122 # macro
CL_ADDRESS_NONE = 0x1130 # macro
CL_ADDRESS_CLAMP_TO_EDGE = 0x1131 # macro
CL_ADDRESS_CLAMP = 0x1132 # macro
CL_ADDRESS_REPEAT = 0x1133 # macro
CL_ADDRESS_MIRRORED_REPEAT = 0x1134 # macro
CL_FILTER_NEAREST = 0x1140 # macro
CL_FILTER_LINEAR = 0x1141 # macro
CL_SAMPLER_REFERENCE_COUNT = 0x1150 # macro
CL_SAMPLER_CONTEXT = 0x1151 # macro
CL_SAMPLER_NORMALIZED_COORDS = 0x1152 # macro
CL_SAMPLER_ADDRESSING_MODE = 0x1153 # macro
CL_SAMPLER_FILTER_MODE = 0x1154 # macro
CL_SAMPLER_MIP_FILTER_MODE = 0x1155 # macro
CL_SAMPLER_LOD_MIN = 0x1156 # macro
CL_SAMPLER_LOD_MAX = 0x1157 # macro
CL_SAMPLER_PROPERTIES = 0x1158 # macro
CL_MAP_READ = (1<<0) # macro
CL_MAP_WRITE = (1<<1) # macro
CL_MAP_WRITE_INVALIDATE_REGION = (1<<2) # macro
CL_PROGRAM_REFERENCE_COUNT = 0x1160 # macro
CL_PROGRAM_CONTEXT = 0x1161 # macro
CL_PROGRAM_NUM_DEVICES = 0x1162 # macro
CL_PROGRAM_DEVICES = 0x1163 # macro
CL_PROGRAM_SOURCE = 0x1164 # macro
CL_PROGRAM_BINARY_SIZES = 0x1165 # macro
CL_PROGRAM_BINARIES = 0x1166 # macro
CL_PROGRAM_NUM_KERNELS = 0x1167 # macro
CL_PROGRAM_KERNEL_NAMES = 0x1168 # macro
CL_PROGRAM_IL = 0x1169 # macro
CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116A # macro
CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116B # macro
CL_PROGRAM_BUILD_STATUS = 0x1181 # macro
CL_PROGRAM_BUILD_OPTIONS = 0x1182 # macro
CL_PROGRAM_BUILD_LOG = 0x1183 # macro
CL_PROGRAM_BINARY_TYPE = 0x1184 # macro
CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185 # macro
CL_PROGRAM_BINARY_TYPE_NONE = 0x0 # macro
CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1 # macro
CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x2 # macro
CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4 # macro
CL_BUILD_SUCCESS = 0 # macro
CL_BUILD_NONE = -1 # macro
CL_BUILD_ERROR = -2 # macro
CL_BUILD_IN_PROGRESS = -3 # macro
CL_KERNEL_FUNCTION_NAME = 0x1190 # macro
CL_KERNEL_NUM_ARGS = 0x1191 # macro
CL_KERNEL_REFERENCE_COUNT = 0x1192 # macro
CL_KERNEL_CONTEXT = 0x1193 # macro
CL_KERNEL_PROGRAM = 0x1194 # macro
CL_KERNEL_ATTRIBUTES = 0x1195 # macro
CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196 # macro
CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197 # macro
CL_KERNEL_ARG_TYPE_NAME = 0x1198 # macro
CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199 # macro
CL_KERNEL_ARG_NAME = 0x119A # macro
CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119B # macro
CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119C # macro
CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119D # macro
CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119E # macro
CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0 # macro
CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1 # macro
CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2 # macro
CL_KERNEL_ARG_ACCESS_NONE = 0x11A3 # macro
CL_KERNEL_ARG_TYPE_NONE = 0 # macro
CL_KERNEL_ARG_TYPE_CONST = (1<<0) # macro
CL_KERNEL_ARG_TYPE_RESTRICT = (1<<1) # macro
CL_KERNEL_ARG_TYPE_VOLATILE = (1<<2) # macro
CL_KERNEL_ARG_TYPE_PIPE = (1<<3) # macro
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0 # macro
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1 # macro
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2 # macro
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3 # macro
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4 # macro
CL_KERNEL_GLOBAL_WORK_SIZE = 0x11B5 # macro
CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033 # macro
CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034 # macro
CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11B8 # macro
CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11B9 # macro
CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11BA # macro
CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6 # macro
CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11B7 # macro
CL_EVENT_COMMAND_QUEUE = 0x11D0 # macro
CL_EVENT_COMMAND_TYPE = 0x11D1 # macro
CL_EVENT_REFERENCE_COUNT = 0x11D2 # macro
CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3 # macro
CL_EVENT_CONTEXT = 0x11D4 # macro
CL_COMMAND_NDRANGE_KERNEL = 0x11F0 # macro
CL_COMMAND_TASK = 0x11F1 # macro
CL_COMMAND_NATIVE_KERNEL = 0x11F2 # macro
CL_COMMAND_READ_BUFFER = 0x11F3 # macro
CL_COMMAND_WRITE_BUFFER = 0x11F4 # macro
CL_COMMAND_COPY_BUFFER = 0x11F5 # macro
CL_COMMAND_READ_IMAGE = 0x11F6 # macro
CL_COMMAND_WRITE_IMAGE = 0x11F7 # macro
CL_COMMAND_COPY_IMAGE = 0x11F8 # macro
CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9 # macro
CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA # macro
CL_COMMAND_MAP_BUFFER = 0x11FB # macro
CL_COMMAND_MAP_IMAGE = 0x11FC # macro
CL_COMMAND_UNMAP_MEM_OBJECT = 0x11FD # macro
CL_COMMAND_MARKER = 0x11FE # macro
CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF # macro
CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200 # macro
CL_COMMAND_READ_BUFFER_RECT = 0x1201 # macro
CL_COMMAND_WRITE_BUFFER_RECT = 0x1202 # macro
CL_COMMAND_COPY_BUFFER_RECT = 0x1203 # macro
CL_COMMAND_USER = 0x1204 # macro
CL_COMMAND_BARRIER = 0x1205 # macro
CL_COMMAND_MIGRATE_MEM_OBJECTS = 0x1206 # macro
CL_COMMAND_FILL_BUFFER = 0x1207 # macro
CL_COMMAND_FILL_IMAGE = 0x1208 # macro
CL_COMMAND_SVM_FREE = 0x1209 # macro
CL_COMMAND_SVM_MEMCPY = 0x120A # macro
CL_COMMAND_SVM_MEMFILL = 0x120B # macro
CL_COMMAND_SVM_MAP = 0x120C # macro
CL_COMMAND_SVM_UNMAP = 0x120D # macro
CL_COMMAND_SVM_MIGRATE_MEM = 0x120E # macro
CL_COMPLETE = 0x0 # macro
CL_RUNNING = 0x1 # macro
CL_SUBMITTED = 0x2 # macro
CL_QUEUED = 0x3 # macro
CL_BUFFER_CREATE_TYPE_REGION = 0x1220 # macro
CL_PROFILING_COMMAND_QUEUED = 0x1280 # macro
CL_PROFILING_COMMAND_SUBMIT = 0x1281 # macro
CL_PROFILING_COMMAND_START = 0x1282 # macro
CL_PROFILING_COMMAND_END = 0x1283 # macro
CL_PROFILING_COMMAND_COMPLETE = 0x1284 # macro
CL_DEVICE_ATOMIC_ORDER_RELAXED = (1<<0) # macro
CL_DEVICE_ATOMIC_ORDER_ACQ_REL = (1<<1) # macro
CL_DEVICE_ATOMIC_ORDER_SEQ_CST = (1<<2) # macro
CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM = (1<<3) # macro
CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP = (1<<4) # macro
CL_DEVICE_ATOMIC_SCOPE_DEVICE = (1<<5) # macro
CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES = (1<<6) # macro
CL_DEVICE_QUEUE_SUPPORTED = (1<<0) # macro
CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT = (1<<1) # macro
CL_KHRONOS_VENDOR_ID_CODEPLAY = 0x10004 # macro
CL_VERSION_MAJOR_BITS = (10) # macro
CL_VERSION_MINOR_BITS = (10) # macro
CL_VERSION_PATCH_BITS = (12) # macro
# CL_VERSION_MAJOR_MASK = ((1<<(10)) # macro
# CL_VERSION_MINOR_MASK = ((1<<(10)) # macro
# CL_VERSION_PATCH_MASK = ((1<<(12)) # macro
# def CL_VERSION_MAJOR(version):  # macro
#    return ((version)>>((10)+(12)))
# def CL_VERSION_MINOR(version):  # macro
#    return (((version)>>(12))&((1<<(10)))
# def CL_VERSION_PATCH(version):  # macro
#    return ((version)&((1<<(12)))
# def CL_MAKE_VERSION(major, minor, patch):  # macro
#    return ((((major)&((1<<(10)))<<((10)+(12)))|(((minor)&((1<<(10)))<<(12))|((patch)&((1<<(12))))
class struct__cl_platform_id(Structure):
    pass

cl_platform_id = ctypes.POINTER(struct__cl_platform_id)
class struct__cl_device_id(Structure):
    pass

cl_device_id = ctypes.POINTER(struct__cl_device_id)
class struct__cl_context(Structure):
    pass

cl_context = ctypes.POINTER(struct__cl_context)
class struct__cl_command_queue(Structure):
    pass

cl_command_queue = ctypes.POINTER(struct__cl_command_queue)
class struct__cl_mem(Structure):
    pass

cl_mem = ctypes.POINTER(struct__cl_mem)
class struct__cl_program(Structure):
    pass

cl_program = ctypes.POINTER(struct__cl_program)
class struct__cl_kernel(Structure):
    pass

cl_kernel = ctypes.POINTER(struct__cl_kernel)
class struct__cl_event(Structure):
    pass

cl_event = ctypes.POINTER(struct__cl_event)
class struct__cl_sampler(Structure):
    pass

cl_sampler = ctypes.POINTER(struct__cl_sampler)
cl_bool = ctypes.c_uint32
cl_bitfield = ctypes.c_uint64
cl_properties = ctypes.c_uint64
cl_device_type = ctypes.c_uint64
cl_platform_info = ctypes.c_uint32
cl_device_info = ctypes.c_uint32
cl_device_fp_config = ctypes.c_uint64
cl_device_mem_cache_type = ctypes.c_uint32
cl_device_local_mem_type = ctypes.c_uint32
cl_device_exec_capabilities = ctypes.c_uint64
cl_device_svm_capabilities = ctypes.c_uint64
cl_command_queue_properties = ctypes.c_uint64
cl_device_partition_property = ctypes.c_int64
cl_device_affinity_domain = ctypes.c_uint64
cl_context_properties = ctypes.c_int64
cl_context_info = ctypes.c_uint32
cl_queue_properties = ctypes.c_uint64
cl_command_queue_info = ctypes.c_uint32
cl_channel_order = ctypes.c_uint32
cl_channel_type = ctypes.c_uint32
cl_mem_flags = ctypes.c_uint64
cl_svm_mem_flags = ctypes.c_uint64
cl_mem_object_type = ctypes.c_uint32
cl_mem_info = ctypes.c_uint32
cl_mem_migration_flags = ctypes.c_uint64
cl_image_info = ctypes.c_uint32
cl_buffer_create_type = ctypes.c_uint32
cl_addressing_mode = ctypes.c_uint32
cl_filter_mode = ctypes.c_uint32
cl_sampler_info = ctypes.c_uint32
cl_map_flags = ctypes.c_uint64
cl_pipe_properties = ctypes.c_int64
cl_pipe_info = ctypes.c_uint32
cl_program_info = ctypes.c_uint32
cl_program_build_info = ctypes.c_uint32
cl_program_binary_type = ctypes.c_uint32
cl_build_status = ctypes.c_int32
cl_kernel_info = ctypes.c_uint32
cl_kernel_arg_info = ctypes.c_uint32
cl_kernel_arg_address_qualifier = ctypes.c_uint32
cl_kernel_arg_access_qualifier = ctypes.c_uint32
cl_kernel_arg_type_qualifier = ctypes.c_uint64
cl_kernel_work_group_info = ctypes.c_uint32
cl_kernel_sub_group_info = ctypes.c_uint32
cl_event_info = ctypes.c_uint32
cl_command_type = ctypes.c_uint32
cl_profiling_info = ctypes.c_uint32
cl_sampler_properties = ctypes.c_uint64
cl_kernel_exec_info = ctypes.c_uint32
cl_device_atomic_capabilities = ctypes.c_uint64
cl_device_device_enqueue_capabilities = ctypes.c_uint64
cl_khronos_vendor_id = ctypes.c_uint32
cl_mem_properties = ctypes.c_uint64
cl_version = ctypes.c_uint32
class struct__cl_image_format(Structure):
    pass

struct__cl_image_format._pack_ = 1 # source:False
struct__cl_image_format._fields_ = [
    ('image_channel_order', ctypes.c_uint32),
    ('image_channel_data_type', ctypes.c_uint32),
]

cl_image_format = struct__cl_image_format
class struct__cl_image_desc(Structure):
    pass

class union__cl_image_desc_0(Union):
    pass

union__cl_image_desc_0._pack_ = 1 # source:False
union__cl_image_desc_0._fields_ = [
    ('buffer', ctypes.POINTER(struct__cl_mem)),
    ('mem_object', ctypes.POINTER(struct__cl_mem)),
]

struct__cl_image_desc._pack_ = 1 # source:False
struct__cl_image_desc._anonymous_ = ('_0',)
struct__cl_image_desc._fields_ = [
    ('image_type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('image_width', ctypes.c_uint64),
    ('image_height', ctypes.c_uint64),
    ('image_depth', ctypes.c_uint64),
    ('image_array_size', ctypes.c_uint64),
    ('image_row_pitch', ctypes.c_uint64),
    ('image_slice_pitch', ctypes.c_uint64),
    ('num_mip_levels', ctypes.c_uint32),
    ('num_samples', ctypes.c_uint32),
    ('_0', union__cl_image_desc_0),
]

cl_image_desc = struct__cl_image_desc
class struct__cl_buffer_region(Structure):
    pass

struct__cl_buffer_region._pack_ = 1 # source:False
struct__cl_buffer_region._fields_ = [
    ('origin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

cl_buffer_region = struct__cl_buffer_region
class struct__cl_name_version(Structure):
    pass

struct__cl_name_version._pack_ = 1 # source:False
struct__cl_name_version._fields_ = [
    ('version', ctypes.c_uint32),
    ('name', ctypes.c_char * 64),
]

cl_name_version = struct__cl_name_version
cl_int = ctypes.c_int32
cl_uint = ctypes.c_uint32
try:
    clGetPlatformIDs = _libraries['libOpenCL.so.1'].clGetPlatformIDs
    clGetPlatformIDs.restype = cl_int
    clGetPlatformIDs.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_platform_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    clGetPlatformInfo = _libraries['libOpenCL.so.1'].clGetPlatformInfo
    clGetPlatformInfo.restype = cl_int
    clGetPlatformInfo.argtypes = [cl_platform_id, cl_platform_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetDeviceIDs = _libraries['libOpenCL.so.1'].clGetDeviceIDs
    clGetDeviceIDs.restype = cl_int
    clGetDeviceIDs.argtypes = [cl_platform_id, cl_device_type, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetDeviceInfo = _libraries['libOpenCL.so.1'].clGetDeviceInfo
    clGetDeviceInfo.restype = cl_int
    clGetDeviceInfo.argtypes = [cl_device_id, cl_device_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateSubDevices = _libraries['libOpenCL.so.1'].clCreateSubDevices
    clCreateSubDevices.restype = cl_int
    clCreateSubDevices.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clRetainDevice = _libraries['libOpenCL.so.1'].clRetainDevice
    clRetainDevice.restype = cl_int
    clRetainDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clReleaseDevice = _libraries['libOpenCL.so.1'].clReleaseDevice
    clReleaseDevice.restype = cl_int
    clReleaseDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clSetDefaultDeviceCommandQueue = _libraries['libOpenCL.so.1'].clSetDefaultDeviceCommandQueue
    clSetDefaultDeviceCommandQueue.restype = cl_int
    clSetDefaultDeviceCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue]
except AttributeError:
    pass
try:
    clGetDeviceAndHostTimer = _libraries['libOpenCL.so.1'].clGetDeviceAndHostTimer
    clGetDeviceAndHostTimer.restype = cl_int
    clGetDeviceAndHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetHostTimer = _libraries['libOpenCL.so.1'].clGetHostTimer
    clGetHostTimer.restype = cl_int
    clGetHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateContext = _libraries['libOpenCL.so.1'].clCreateContext
    clCreateContext.restype = cl_context
    clCreateContext.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateContextFromType = _libraries['libOpenCL.so.1'].clCreateContextFromType
    clCreateContextFromType.restype = cl_context
    clCreateContextFromType.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_device_type, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainContext = _libraries['libOpenCL.so.1'].clRetainContext
    clRetainContext.restype = cl_int
    clRetainContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clReleaseContext = _libraries['libOpenCL.so.1'].clReleaseContext
    clReleaseContext.restype = cl_int
    clReleaseContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clGetContextInfo = _libraries['libOpenCL.so.1'].clGetContextInfo
    clGetContextInfo.restype = cl_int
    clGetContextInfo.argtypes = [cl_context, cl_context_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetContextDestructorCallback = _libraries['libOpenCL.so.1'].clSetContextDestructorCallback
    clSetContextDestructorCallback.restype = cl_int
    clSetContextDestructorCallback.argtypes = [cl_context, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_context), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateCommandQueueWithProperties = _libraries['libOpenCL.so.1'].clCreateCommandQueueWithProperties
    clCreateCommandQueueWithProperties.restype = cl_command_queue
    clCreateCommandQueueWithProperties.argtypes = [cl_context, cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainCommandQueue = _libraries['libOpenCL.so.1'].clRetainCommandQueue
    clRetainCommandQueue.restype = cl_int
    clRetainCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clReleaseCommandQueue = _libraries['libOpenCL.so.1'].clReleaseCommandQueue
    clReleaseCommandQueue.restype = cl_int
    clReleaseCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clGetCommandQueueInfo = _libraries['libOpenCL.so.1'].clGetCommandQueueInfo
    clGetCommandQueueInfo.restype = cl_int
    clGetCommandQueueInfo.argtypes = [cl_command_queue, cl_command_queue_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateBuffer = _libraries['libOpenCL.so.1'].clCreateBuffer
    clCreateBuffer.restype = cl_mem
    clCreateBuffer.argtypes = [cl_context, cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSubBuffer = _libraries['libOpenCL.so.1'].clCreateSubBuffer
    clCreateSubBuffer.restype = cl_mem
    clCreateSubBuffer.argtypes = [cl_mem, cl_mem_flags, cl_buffer_create_type, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage = _libraries['libOpenCL.so.1'].clCreateImage
    clCreateImage.restype = cl_mem
    clCreateImage.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreatePipe = _libraries['libOpenCL.so.1'].clCreatePipe
    clCreatePipe.restype = cl_mem
    clCreatePipe.argtypes = [cl_context, cl_mem_flags, cl_uint, cl_uint, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateBufferWithProperties = _libraries['libOpenCL.so.1'].clCreateBufferWithProperties
    clCreateBufferWithProperties.restype = cl_mem
    clCreateBufferWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImageWithProperties = _libraries['libOpenCL.so.1'].clCreateImageWithProperties
    clCreateImageWithProperties.restype = cl_mem
    clCreateImageWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainMemObject = _libraries['libOpenCL.so.1'].clRetainMemObject
    clRetainMemObject.restype = cl_int
    clRetainMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clReleaseMemObject = _libraries['libOpenCL.so.1'].clReleaseMemObject
    clReleaseMemObject.restype = cl_int
    clReleaseMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clGetSupportedImageFormats = _libraries['libOpenCL.so.1'].clGetSupportedImageFormats
    clGetSupportedImageFormats.restype = cl_int
    clGetSupportedImageFormats.argtypes = [cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetMemObjectInfo = _libraries['libOpenCL.so.1'].clGetMemObjectInfo
    clGetMemObjectInfo.restype = cl_int
    clGetMemObjectInfo.argtypes = [cl_mem, cl_mem_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetImageInfo = _libraries['libOpenCL.so.1'].clGetImageInfo
    clGetImageInfo.restype = cl_int
    clGetImageInfo.argtypes = [cl_mem, cl_image_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetPipeInfo = _libraries['libOpenCL.so.1'].clGetPipeInfo
    clGetPipeInfo.restype = cl_int
    clGetPipeInfo.argtypes = [cl_mem, cl_pipe_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetMemObjectDestructorCallback = _libraries['libOpenCL.so.1'].clSetMemObjectDestructorCallback
    clSetMemObjectDestructorCallback.restype = cl_int
    clSetMemObjectDestructorCallback.argtypes = [cl_mem, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_mem), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSVMAlloc = _libraries['libOpenCL.so.1'].clSVMAlloc
    clSVMAlloc.restype = ctypes.POINTER(None)
    clSVMAlloc.argtypes = [cl_context, cl_svm_mem_flags, size_t, cl_uint]
except AttributeError:
    pass
try:
    clSVMFree = _libraries['libOpenCL.so.1'].clSVMFree
    clSVMFree.restype = None
    clSVMFree.argtypes = [cl_context, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateSamplerWithProperties = _libraries['libOpenCL.so.1'].clCreateSamplerWithProperties
    clCreateSamplerWithProperties.restype = cl_sampler
    clCreateSamplerWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainSampler = _libraries['libOpenCL.so.1'].clRetainSampler
    clRetainSampler.restype = cl_int
    clRetainSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clReleaseSampler = _libraries['libOpenCL.so.1'].clReleaseSampler
    clReleaseSampler.restype = cl_int
    clReleaseSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clGetSamplerInfo = _libraries['libOpenCL.so.1'].clGetSamplerInfo
    clGetSamplerInfo.restype = cl_int
    clGetSamplerInfo.argtypes = [cl_sampler, cl_sampler_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateProgramWithSource = _libraries['libOpenCL.so.1'].clCreateProgramWithSource
    clCreateProgramWithSource.restype = cl_program
    clCreateProgramWithSource.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBinary = _libraries['libOpenCL.so.1'].clCreateProgramWithBinary
    clCreateProgramWithBinary.restype = cl_program
    clCreateProgramWithBinary.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBuiltInKernels = _libraries['libOpenCL.so.1'].clCreateProgramWithBuiltInKernels
    clCreateProgramWithBuiltInKernels.restype = cl_program
    clCreateProgramWithBuiltInKernels.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithIL = _libraries['libOpenCL.so.1'].clCreateProgramWithIL
    clCreateProgramWithIL.restype = cl_program
    clCreateProgramWithIL.argtypes = [cl_context, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainProgram = _libraries['libOpenCL.so.1'].clRetainProgram
    clRetainProgram.restype = cl_int
    clRetainProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clReleaseProgram = _libraries['libOpenCL.so.1'].clReleaseProgram
    clReleaseProgram.restype = cl_int
    clReleaseProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clBuildProgram = _libraries['libOpenCL.so.1'].clBuildProgram
    clBuildProgram.restype = cl_int
    clBuildProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCompileProgram = _libraries['libOpenCL.so.1'].clCompileProgram
    clCompileProgram.restype = cl_int
    clCompileProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clLinkProgram = _libraries['libOpenCL.so.1'].clLinkProgram
    clLinkProgram.restype = cl_program
    clLinkProgram.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clSetProgramReleaseCallback = _libraries['libOpenCL.so.1'].clSetProgramReleaseCallback
    clSetProgramReleaseCallback.restype = cl_int
    clSetProgramReleaseCallback.argtypes = [cl_program, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetProgramSpecializationConstant = _libraries['libOpenCL.so.1'].clSetProgramSpecializationConstant
    clSetProgramSpecializationConstant.restype = cl_int
    clSetProgramSpecializationConstant.argtypes = [cl_program, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clUnloadPlatformCompiler = _libraries['libOpenCL.so.1'].clUnloadPlatformCompiler
    clUnloadPlatformCompiler.restype = cl_int
    clUnloadPlatformCompiler.argtypes = [cl_platform_id]
except AttributeError:
    pass
try:
    clGetProgramInfo = _libraries['libOpenCL.so.1'].clGetProgramInfo
    clGetProgramInfo.restype = cl_int
    clGetProgramInfo.argtypes = [cl_program, cl_program_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetProgramBuildInfo = _libraries['libOpenCL.so.1'].clGetProgramBuildInfo
    clGetProgramBuildInfo.restype = cl_int
    clGetProgramBuildInfo.argtypes = [cl_program, cl_device_id, cl_program_build_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateKernel = _libraries['libOpenCL.so.1'].clCreateKernel
    clCreateKernel.restype = cl_kernel
    clCreateKernel.argtypes = [cl_program, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateKernelsInProgram = _libraries['libOpenCL.so.1'].clCreateKernelsInProgram
    clCreateKernelsInProgram.restype = cl_int
    clCreateKernelsInProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_kernel)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clCloneKernel = _libraries['libOpenCL.so.1'].clCloneKernel
    clCloneKernel.restype = cl_kernel
    clCloneKernel.argtypes = [cl_kernel, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainKernel = _libraries['libOpenCL.so.1'].clRetainKernel
    clRetainKernel.restype = cl_int
    clRetainKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clReleaseKernel = _libraries['libOpenCL.so.1'].clReleaseKernel
    clReleaseKernel.restype = cl_int
    clReleaseKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clSetKernelArg = _libraries['libOpenCL.so.1'].clSetKernelArg
    clSetKernelArg.restype = cl_int
    clSetKernelArg.argtypes = [cl_kernel, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelArgSVMPointer = _libraries['libOpenCL.so.1'].clSetKernelArgSVMPointer
    clSetKernelArgSVMPointer.restype = cl_int
    clSetKernelArgSVMPointer.argtypes = [cl_kernel, cl_uint, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelExecInfo = _libraries['libOpenCL.so.1'].clSetKernelExecInfo
    clSetKernelExecInfo.restype = cl_int
    clSetKernelExecInfo.argtypes = [cl_kernel, cl_kernel_exec_info, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetKernelInfo = _libraries['libOpenCL.so.1'].clGetKernelInfo
    clGetKernelInfo.restype = cl_int
    clGetKernelInfo.argtypes = [cl_kernel, cl_kernel_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelArgInfo = _libraries['libOpenCL.so.1'].clGetKernelArgInfo
    clGetKernelArgInfo.restype = cl_int
    clGetKernelArgInfo.argtypes = [cl_kernel, cl_uint, cl_kernel_arg_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelWorkGroupInfo = _libraries['libOpenCL.so.1'].clGetKernelWorkGroupInfo
    clGetKernelWorkGroupInfo.restype = cl_int
    clGetKernelWorkGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelSubGroupInfo = _libraries['libOpenCL.so.1'].clGetKernelSubGroupInfo
    clGetKernelSubGroupInfo.restype = cl_int
    clGetKernelSubGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, ctypes.POINTER(None), size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clWaitForEvents = _libraries['libOpenCL.so.1'].clWaitForEvents
    clWaitForEvents.restype = cl_int
    clWaitForEvents.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetEventInfo = _libraries['libOpenCL.so.1'].clGetEventInfo
    clGetEventInfo.restype = cl_int
    clGetEventInfo.argtypes = [cl_event, cl_event_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateUserEvent = _libraries['libOpenCL.so.1'].clCreateUserEvent
    clCreateUserEvent.restype = cl_event
    clCreateUserEvent.argtypes = [cl_context, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainEvent = _libraries['libOpenCL.so.1'].clRetainEvent
    clRetainEvent.restype = cl_int
    clRetainEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clReleaseEvent = _libraries['libOpenCL.so.1'].clReleaseEvent
    clReleaseEvent.restype = cl_int
    clReleaseEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clSetUserEventStatus = _libraries['libOpenCL.so.1'].clSetUserEventStatus
    clSetUserEventStatus.restype = cl_int
    clSetUserEventStatus.argtypes = [cl_event, cl_int]
except AttributeError:
    pass
try:
    clSetEventCallback = _libraries['libOpenCL.so.1'].clSetEventCallback
    clSetEventCallback.restype = cl_int
    clSetEventCallback.argtypes = [cl_event, cl_int, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_event), ctypes.c_int32, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetEventProfilingInfo = _libraries['libOpenCL.so.1'].clGetEventProfilingInfo
    clGetEventProfilingInfo.restype = cl_int
    clGetEventProfilingInfo.argtypes = [cl_event, cl_profiling_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clFlush = _libraries['libOpenCL.so.1'].clFlush
    clFlush.restype = cl_int
    clFlush.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clFinish = _libraries['libOpenCL.so.1'].clFinish
    clFinish.restype = cl_int
    clFinish.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clEnqueueReadBuffer = _libraries['libOpenCL.so.1'].clEnqueueReadBuffer
    clEnqueueReadBuffer.restype = cl_int
    clEnqueueReadBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadBufferRect = _libraries['libOpenCL.so.1'].clEnqueueReadBufferRect
    clEnqueueReadBufferRect.restype = cl_int
    clEnqueueReadBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBuffer = _libraries['libOpenCL.so.1'].clEnqueueWriteBuffer
    clEnqueueWriteBuffer.restype = cl_int
    clEnqueueWriteBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBufferRect = _libraries['libOpenCL.so.1'].clEnqueueWriteBufferRect
    clEnqueueWriteBufferRect.restype = cl_int
    clEnqueueWriteBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillBuffer = _libraries['libOpenCL.so.1'].clEnqueueFillBuffer
    clEnqueueFillBuffer.restype = cl_int
    clEnqueueFillBuffer.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBuffer = _libraries['libOpenCL.so.1'].clEnqueueCopyBuffer
    clEnqueueCopyBuffer.restype = cl_int
    clEnqueueCopyBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferRect = _libraries['libOpenCL.so.1'].clEnqueueCopyBufferRect
    clEnqueueCopyBufferRect.restype = cl_int
    clEnqueueCopyBufferRect.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadImage = _libraries['libOpenCL.so.1'].clEnqueueReadImage
    clEnqueueReadImage.restype = cl_int
    clEnqueueReadImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteImage = _libraries['libOpenCL.so.1'].clEnqueueWriteImage
    clEnqueueWriteImage.restype = cl_int
    clEnqueueWriteImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillImage = _libraries['libOpenCL.so.1'].clEnqueueFillImage
    clEnqueueFillImage.restype = cl_int
    clEnqueueFillImage.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImage = _libraries['libOpenCL.so.1'].clEnqueueCopyImage
    clEnqueueCopyImage.restype = cl_int
    clEnqueueCopyImage.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImageToBuffer = _libraries['libOpenCL.so.1'].clEnqueueCopyImageToBuffer
    clEnqueueCopyImageToBuffer.restype = cl_int
    clEnqueueCopyImageToBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferToImage = _libraries['libOpenCL.so.1'].clEnqueueCopyBufferToImage
    clEnqueueCopyBufferToImage.restype = cl_int
    clEnqueueCopyBufferToImage.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMapBuffer = _libraries['libOpenCL.so.1'].clEnqueueMapBuffer
    clEnqueueMapBuffer.restype = ctypes.POINTER(None)
    clEnqueueMapBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMapImage = _libraries['libOpenCL.so.1'].clEnqueueMapImage
    clEnqueueMapImage.restype = ctypes.POINTER(None)
    clEnqueueMapImage.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueUnmapMemObject = _libraries['libOpenCL.so.1'].clEnqueueUnmapMemObject
    clEnqueueUnmapMemObject.restype = cl_int
    clEnqueueUnmapMemObject.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMigrateMemObjects = _libraries['libOpenCL.so.1'].clEnqueueMigrateMemObjects
    clEnqueueMigrateMemObjects.restype = cl_int
    clEnqueueMigrateMemObjects.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNDRangeKernel = _libraries['libOpenCL.so.1'].clEnqueueNDRangeKernel
    clEnqueueNDRangeKernel.restype = cl_int
    clEnqueueNDRangeKernel.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNativeKernel = _libraries['libOpenCL.so.1'].clEnqueueNativeKernel
    clEnqueueNativeKernel.restype = cl_int
    clEnqueueNativeKernel.argtypes = [cl_command_queue, ctypes.CFUNCTYPE(None, ctypes.POINTER(None)), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), ctypes.POINTER(ctypes.POINTER(None)), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMarkerWithWaitList = _libraries['libOpenCL.so.1'].clEnqueueMarkerWithWaitList
    clEnqueueMarkerWithWaitList.restype = cl_int
    clEnqueueMarkerWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrierWithWaitList = _libraries['libOpenCL.so.1'].clEnqueueBarrierWithWaitList
    clEnqueueBarrierWithWaitList.restype = cl_int
    clEnqueueBarrierWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMFree = _libraries['libOpenCL.so.1'].clEnqueueSVMFree
    clEnqueueSVMFree.restype = cl_int
    clEnqueueSVMFree.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(None) * 0, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_command_queue), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(None)), ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemcpy = _libraries['libOpenCL.so.1'].clEnqueueSVMMemcpy
    clEnqueueSVMMemcpy.restype = cl_int
    clEnqueueSVMMemcpy.argtypes = [cl_command_queue, cl_bool, ctypes.POINTER(None), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemFill = _libraries['libOpenCL.so.1'].clEnqueueSVMMemFill
    clEnqueueSVMMemFill.restype = cl_int
    clEnqueueSVMMemFill.argtypes = [cl_command_queue, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMap = _libraries['libOpenCL.so.1'].clEnqueueSVMMap
    clEnqueueSVMMap.restype = cl_int
    clEnqueueSVMMap.argtypes = [cl_command_queue, cl_bool, cl_map_flags, ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMUnmap = _libraries['libOpenCL.so.1'].clEnqueueSVMUnmap
    clEnqueueSVMUnmap.restype = cl_int
    clEnqueueSVMUnmap.argtypes = [cl_command_queue, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMigrateMem = _libraries['libOpenCL.so.1'].clEnqueueSVMMigrateMem
    clEnqueueSVMMigrateMem.restype = cl_int
    clEnqueueSVMMigrateMem.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddressForPlatform = _libraries['libOpenCL.so.1'].clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddressForPlatform.argtypes = [cl_platform_id, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateImage2D = _libraries['libOpenCL.so.1'].clCreateImage2D
    clCreateImage2D.restype = cl_mem
    clCreateImage2D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage3D = _libraries['libOpenCL.so.1'].clCreateImage3D
    clCreateImage3D.restype = cl_mem
    clCreateImage3D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMarker = _libraries['libOpenCL.so.1'].clEnqueueMarker
    clEnqueueMarker.restype = cl_int
    clEnqueueMarker.argtypes = [cl_command_queue, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWaitForEvents = _libraries['libOpenCL.so.1'].clEnqueueWaitForEvents
    clEnqueueWaitForEvents.restype = cl_int
    clEnqueueWaitForEvents.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrier = _libraries['libOpenCL.so.1'].clEnqueueBarrier
    clEnqueueBarrier.restype = cl_int
    clEnqueueBarrier.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clUnloadCompiler = _libraries['libOpenCL.so.1'].clUnloadCompiler
    clUnloadCompiler.restype = cl_int
    clUnloadCompiler.argtypes = []
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddress = _libraries['libOpenCL.so.1'].clGetExtensionFunctionAddress
    clGetExtensionFunctionAddress.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddress.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateCommandQueue = _libraries['libOpenCL.so.1'].clCreateCommandQueue
    clCreateCommandQueue.restype = cl_command_queue
    clCreateCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue_properties, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSampler = _libraries['libOpenCL.so.1'].clCreateSampler
    clCreateSampler.restype = cl_sampler
    clCreateSampler.argtypes = [cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueTask = _libraries['libOpenCL.so.1'].clEnqueueTask
    clEnqueueTask.restype = cl_int
    clEnqueueTask.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
__all__ = \
    ['CL_A', 'CL_ABGR', 'CL_ADDRESS_CLAMP',
    'CL_ADDRESS_CLAMP_TO_EDGE', 'CL_ADDRESS_MIRRORED_REPEAT',
    'CL_ADDRESS_NONE', 'CL_ADDRESS_REPEAT', 'CL_ARGB', 'CL_BGRA',
    'CL_BLOCKING', 'CL_BUFFER_CREATE_TYPE_REGION', 'CL_BUILD_ERROR',
    'CL_BUILD_IN_PROGRESS', 'CL_BUILD_NONE',
    'CL_BUILD_PROGRAM_FAILURE', 'CL_BUILD_SUCCESS',
    'CL_COMMAND_ACQUIRE_GL_OBJECTS', 'CL_COMMAND_BARRIER',
    'CL_COMMAND_COPY_BUFFER', 'CL_COMMAND_COPY_BUFFER_RECT',
    'CL_COMMAND_COPY_BUFFER_TO_IMAGE', 'CL_COMMAND_COPY_IMAGE',
    'CL_COMMAND_COPY_IMAGE_TO_BUFFER', 'CL_COMMAND_FILL_BUFFER',
    'CL_COMMAND_FILL_IMAGE', 'CL_COMMAND_MAP_BUFFER',
    'CL_COMMAND_MAP_IMAGE', 'CL_COMMAND_MARKER',
    'CL_COMMAND_MIGRATE_MEM_OBJECTS', 'CL_COMMAND_NATIVE_KERNEL',
    'CL_COMMAND_NDRANGE_KERNEL', 'CL_COMMAND_READ_BUFFER',
    'CL_COMMAND_READ_BUFFER_RECT', 'CL_COMMAND_READ_IMAGE',
    'CL_COMMAND_RELEASE_GL_OBJECTS', 'CL_COMMAND_SVM_FREE',
    'CL_COMMAND_SVM_MAP', 'CL_COMMAND_SVM_MEMCPY',
    'CL_COMMAND_SVM_MEMFILL', 'CL_COMMAND_SVM_MIGRATE_MEM',
    'CL_COMMAND_SVM_UNMAP', 'CL_COMMAND_TASK',
    'CL_COMMAND_UNMAP_MEM_OBJECT', 'CL_COMMAND_USER',
    'CL_COMMAND_WRITE_BUFFER', 'CL_COMMAND_WRITE_BUFFER_RECT',
    'CL_COMMAND_WRITE_IMAGE', 'CL_COMPILER_NOT_AVAILABLE',
    'CL_COMPILE_PROGRAM_FAILURE', 'CL_COMPLETE', 'CL_CONTEXT_DEVICES',
    'CL_CONTEXT_INTEROP_USER_SYNC', 'CL_CONTEXT_NUM_DEVICES',
    'CL_CONTEXT_PLATFORM', 'CL_CONTEXT_PROPERTIES',
    'CL_CONTEXT_REFERENCE_COUNT', 'CL_DEPTH', 'CL_DEPTH_STENCIL',
    'CL_DEVICE_ADDRESS_BITS', 'CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE',
    'CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE',
    'CL_DEVICE_AFFINITY_DOMAIN_NUMA',
    'CL_DEVICE_ATOMIC_FENCE_CAPABILITIES',
    'CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES',
    'CL_DEVICE_ATOMIC_ORDER_ACQ_REL',
    'CL_DEVICE_ATOMIC_ORDER_RELAXED',
    'CL_DEVICE_ATOMIC_ORDER_SEQ_CST',
    'CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES',
    'CL_DEVICE_ATOMIC_SCOPE_DEVICE',
    'CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP',
    'CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM', 'CL_DEVICE_AVAILABLE',
    'CL_DEVICE_BUILT_IN_KERNELS',
    'CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION',
    'CL_DEVICE_COMPILER_AVAILABLE',
    'CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES',
    'CL_DEVICE_DOUBLE_FP_CONFIG', 'CL_DEVICE_ENDIAN_LITTLE',
    'CL_DEVICE_ERROR_CORRECTION_SUPPORT',
    'CL_DEVICE_EXECUTION_CAPABILITIES', 'CL_DEVICE_EXTENSIONS',
    'CL_DEVICE_EXTENSIONS_WITH_VERSION',
    'CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT',
    'CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE',
    'CL_DEVICE_GLOBAL_MEM_CACHE_SIZE',
    'CL_DEVICE_GLOBAL_MEM_CACHE_TYPE', 'CL_DEVICE_GLOBAL_MEM_SIZE',
    'CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE',
    'CL_DEVICE_HOST_UNIFIED_MEMORY', 'CL_DEVICE_ILS_WITH_VERSION',
    'CL_DEVICE_IL_VERSION', 'CL_DEVICE_IMAGE2D_MAX_HEIGHT',
    'CL_DEVICE_IMAGE2D_MAX_WIDTH', 'CL_DEVICE_IMAGE3D_MAX_DEPTH',
    'CL_DEVICE_IMAGE3D_MAX_HEIGHT', 'CL_DEVICE_IMAGE3D_MAX_WIDTH',
    'CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT',
    'CL_DEVICE_IMAGE_MAX_ARRAY_SIZE',
    'CL_DEVICE_IMAGE_MAX_BUFFER_SIZE',
    'CL_DEVICE_IMAGE_PITCH_ALIGNMENT', 'CL_DEVICE_IMAGE_SUPPORT',
    'CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED',
    'CL_DEVICE_LINKER_AVAILABLE', 'CL_DEVICE_LOCAL_MEM_SIZE',
    'CL_DEVICE_LOCAL_MEM_TYPE', 'CL_DEVICE_MAX_CLOCK_FREQUENCY',
    'CL_DEVICE_MAX_COMPUTE_UNITS', 'CL_DEVICE_MAX_CONSTANT_ARGS',
    'CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE',
    'CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE',
    'CL_DEVICE_MAX_MEM_ALLOC_SIZE', 'CL_DEVICE_MAX_NUM_SUB_GROUPS',
    'CL_DEVICE_MAX_ON_DEVICE_EVENTS',
    'CL_DEVICE_MAX_ON_DEVICE_QUEUES', 'CL_DEVICE_MAX_PARAMETER_SIZE',
    'CL_DEVICE_MAX_PIPE_ARGS', 'CL_DEVICE_MAX_READ_IMAGE_ARGS',
    'CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS', 'CL_DEVICE_MAX_SAMPLERS',
    'CL_DEVICE_MAX_WORK_GROUP_SIZE',
    'CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS',
    'CL_DEVICE_MAX_WORK_ITEM_SIZES', 'CL_DEVICE_MAX_WRITE_IMAGE_ARGS',
    'CL_DEVICE_MEM_BASE_ADDR_ALIGN',
    'CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE', 'CL_DEVICE_NAME',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_INT',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG',
    'CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT',
    'CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT',
    'CL_DEVICE_NOT_AVAILABLE', 'CL_DEVICE_NOT_FOUND',
    'CL_DEVICE_NUMERIC_VERSION', 'CL_DEVICE_OPENCL_C_ALL_VERSIONS',
    'CL_DEVICE_OPENCL_C_FEATURES', 'CL_DEVICE_OPENCL_C_VERSION',
    'CL_DEVICE_PARENT_DEVICE', 'CL_DEVICE_PARTITION_AFFINITY_DOMAIN',
    'CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN',
    'CL_DEVICE_PARTITION_BY_COUNTS',
    'CL_DEVICE_PARTITION_BY_COUNTS_LIST_END',
    'CL_DEVICE_PARTITION_EQUALLY', 'CL_DEVICE_PARTITION_FAILED',
    'CL_DEVICE_PARTITION_MAX_SUB_DEVICES',
    'CL_DEVICE_PARTITION_PROPERTIES', 'CL_DEVICE_PARTITION_TYPE',
    'CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS',
    'CL_DEVICE_PIPE_MAX_PACKET_SIZE', 'CL_DEVICE_PIPE_SUPPORT',
    'CL_DEVICE_PLATFORM',
    'CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_INTEROP_USER_SYNC',
    'CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG',
    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT',
    'CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
    'CL_DEVICE_PRINTF_BUFFER_SIZE', 'CL_DEVICE_PROFILE',
    'CL_DEVICE_PROFILING_TIMER_RESOLUTION',
    'CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE',
    'CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE',
    'CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES',
    'CL_DEVICE_QUEUE_ON_HOST_PROPERTIES',
    'CL_DEVICE_QUEUE_PROPERTIES',
    'CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT',
    'CL_DEVICE_QUEUE_SUPPORTED', 'CL_DEVICE_REFERENCE_COUNT',
    'CL_DEVICE_SINGLE_FP_CONFIG',
    'CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS',
    'CL_DEVICE_SVM_ATOMICS', 'CL_DEVICE_SVM_CAPABILITIES',
    'CL_DEVICE_SVM_COARSE_GRAIN_BUFFER',
    'CL_DEVICE_SVM_FINE_GRAIN_BUFFER',
    'CL_DEVICE_SVM_FINE_GRAIN_SYSTEM', 'CL_DEVICE_TYPE',
    'CL_DEVICE_TYPE_ACCELERATOR', 'CL_DEVICE_TYPE_ALL',
    'CL_DEVICE_TYPE_CPU', 'CL_DEVICE_TYPE_CUSTOM',
    'CL_DEVICE_TYPE_DEFAULT', 'CL_DEVICE_TYPE_GPU',
    'CL_DEVICE_VENDOR', 'CL_DEVICE_VENDOR_ID', 'CL_DEVICE_VERSION',
    'CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT',
    'CL_DRIVER_VERSION', 'CL_EVENT_COMMAND_EXECUTION_STATUS',
    'CL_EVENT_COMMAND_QUEUE', 'CL_EVENT_COMMAND_TYPE',
    'CL_EVENT_CONTEXT', 'CL_EVENT_REFERENCE_COUNT', 'CL_EXEC_KERNEL',
    'CL_EXEC_NATIVE_KERNEL',
    'CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST', 'CL_FALSE',
    'CL_FILTER_LINEAR', 'CL_FILTER_NEAREST', 'CL_FLOAT',
    'CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT', 'CL_FP_DENORM',
    'CL_FP_FMA', 'CL_FP_INF_NAN', 'CL_FP_ROUND_TO_INF',
    'CL_FP_ROUND_TO_NEAREST', 'CL_FP_ROUND_TO_ZERO',
    'CL_FP_SOFT_FLOAT', 'CL_GLOBAL', 'CL_HALF_FLOAT',
    'CL_IMAGE_ARRAY_SIZE', 'CL_IMAGE_BUFFER', 'CL_IMAGE_DEPTH',
    'CL_IMAGE_ELEMENT_SIZE', 'CL_IMAGE_FORMAT',
    'CL_IMAGE_FORMAT_MISMATCH', 'CL_IMAGE_FORMAT_NOT_SUPPORTED',
    'CL_IMAGE_HEIGHT', 'CL_IMAGE_NUM_MIP_LEVELS',
    'CL_IMAGE_NUM_SAMPLES', 'CL_IMAGE_ROW_PITCH',
    'CL_IMAGE_SLICE_PITCH', 'CL_IMAGE_WIDTH', 'CL_INTENSITY',
    'CL_INVALID_ARG_INDEX', 'CL_INVALID_ARG_SIZE',
    'CL_INVALID_ARG_VALUE', 'CL_INVALID_BINARY',
    'CL_INVALID_BUFFER_SIZE', 'CL_INVALID_BUILD_OPTIONS',
    'CL_INVALID_COMMAND_QUEUE', 'CL_INVALID_COMPILER_OPTIONS',
    'CL_INVALID_CONTEXT', 'CL_INVALID_DEVICE',
    'CL_INVALID_DEVICE_PARTITION_COUNT', 'CL_INVALID_DEVICE_QUEUE',
    'CL_INVALID_DEVICE_TYPE', 'CL_INVALID_EVENT',
    'CL_INVALID_EVENT_WAIT_LIST', 'CL_INVALID_GLOBAL_OFFSET',
    'CL_INVALID_GLOBAL_WORK_SIZE', 'CL_INVALID_GL_OBJECT',
    'CL_INVALID_HOST_PTR', 'CL_INVALID_IMAGE_DESCRIPTOR',
    'CL_INVALID_IMAGE_FORMAT_DESCRIPTOR', 'CL_INVALID_IMAGE_SIZE',
    'CL_INVALID_KERNEL', 'CL_INVALID_KERNEL_ARGS',
    'CL_INVALID_KERNEL_DEFINITION', 'CL_INVALID_KERNEL_NAME',
    'CL_INVALID_LINKER_OPTIONS', 'CL_INVALID_MEM_OBJECT',
    'CL_INVALID_MIP_LEVEL', 'CL_INVALID_OPERATION',
    'CL_INVALID_PIPE_SIZE', 'CL_INVALID_PLATFORM',
    'CL_INVALID_PROGRAM', 'CL_INVALID_PROGRAM_EXECUTABLE',
    'CL_INVALID_PROPERTY', 'CL_INVALID_QUEUE_PROPERTIES',
    'CL_INVALID_SAMPLER', 'CL_INVALID_SPEC_ID', 'CL_INVALID_VALUE',
    'CL_INVALID_WORK_DIMENSION', 'CL_INVALID_WORK_GROUP_SIZE',
    'CL_INVALID_WORK_ITEM_SIZE', 'CL_KERNEL_ARG_ACCESS_NONE',
    'CL_KERNEL_ARG_ACCESS_QUALIFIER',
    'CL_KERNEL_ARG_ACCESS_READ_ONLY',
    'CL_KERNEL_ARG_ACCESS_READ_WRITE',
    'CL_KERNEL_ARG_ACCESS_WRITE_ONLY',
    'CL_KERNEL_ARG_ADDRESS_CONSTANT', 'CL_KERNEL_ARG_ADDRESS_GLOBAL',
    'CL_KERNEL_ARG_ADDRESS_LOCAL', 'CL_KERNEL_ARG_ADDRESS_PRIVATE',
    'CL_KERNEL_ARG_ADDRESS_QUALIFIER',
    'CL_KERNEL_ARG_INFO_NOT_AVAILABLE', 'CL_KERNEL_ARG_NAME',
    'CL_KERNEL_ARG_TYPE_CONST', 'CL_KERNEL_ARG_TYPE_NAME',
    'CL_KERNEL_ARG_TYPE_NONE', 'CL_KERNEL_ARG_TYPE_PIPE',
    'CL_KERNEL_ARG_TYPE_QUALIFIER', 'CL_KERNEL_ARG_TYPE_RESTRICT',
    'CL_KERNEL_ARG_TYPE_VOLATILE', 'CL_KERNEL_ATTRIBUTES',
    'CL_KERNEL_COMPILE_NUM_SUB_GROUPS',
    'CL_KERNEL_COMPILE_WORK_GROUP_SIZE', 'CL_KERNEL_CONTEXT',
    'CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM',
    'CL_KERNEL_EXEC_INFO_SVM_PTRS', 'CL_KERNEL_FUNCTION_NAME',
    'CL_KERNEL_GLOBAL_WORK_SIZE', 'CL_KERNEL_LOCAL_MEM_SIZE',
    'CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT',
    'CL_KERNEL_MAX_NUM_SUB_GROUPS',
    'CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE', 'CL_KERNEL_NUM_ARGS',
    'CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
    'CL_KERNEL_PRIVATE_MEM_SIZE', 'CL_KERNEL_PROGRAM',
    'CL_KERNEL_REFERENCE_COUNT',
    'CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE',
    'CL_KERNEL_WORK_GROUP_SIZE', 'CL_KHRONOS_VENDOR_ID_CODEPLAY',
    'CL_LINKER_NOT_AVAILABLE', 'CL_LINK_PROGRAM_FAILURE', 'CL_LOCAL',
    'CL_LUMINANCE', 'CL_MAP_FAILURE', 'CL_MAP_READ', 'CL_MAP_WRITE',
    'CL_MAP_WRITE_INVALIDATE_REGION',
    'CL_MAX_SIZE_RESTRICTION_EXCEEDED', 'CL_MEM_ALLOC_HOST_PTR',
    'CL_MEM_ASSOCIATED_MEMOBJECT', 'CL_MEM_CONTEXT',
    'CL_MEM_COPY_HOST_PTR', 'CL_MEM_COPY_OVERLAP', 'CL_MEM_FLAGS',
    'CL_MEM_HOST_NO_ACCESS', 'CL_MEM_HOST_PTR',
    'CL_MEM_HOST_READ_ONLY', 'CL_MEM_HOST_WRITE_ONLY',
    'CL_MEM_KERNEL_READ_AND_WRITE', 'CL_MEM_MAP_COUNT',
    'CL_MEM_OBJECT_ALLOCATION_FAILURE', 'CL_MEM_OBJECT_BUFFER',
    'CL_MEM_OBJECT_IMAGE1D', 'CL_MEM_OBJECT_IMAGE1D_ARRAY',
    'CL_MEM_OBJECT_IMAGE1D_BUFFER', 'CL_MEM_OBJECT_IMAGE2D',
    'CL_MEM_OBJECT_IMAGE2D_ARRAY', 'CL_MEM_OBJECT_IMAGE3D',
    'CL_MEM_OBJECT_PIPE', 'CL_MEM_OFFSET', 'CL_MEM_PROPERTIES',
    'CL_MEM_READ_ONLY', 'CL_MEM_READ_WRITE', 'CL_MEM_REFERENCE_COUNT',
    'CL_MEM_SIZE', 'CL_MEM_SVM_ATOMICS',
    'CL_MEM_SVM_FINE_GRAIN_BUFFER', 'CL_MEM_TYPE',
    'CL_MEM_USES_SVM_POINTER', 'CL_MEM_USE_HOST_PTR',
    'CL_MEM_WRITE_ONLY', 'CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED',
    'CL_MIGRATE_MEM_OBJECT_HOST', 'CL_MISALIGNED_SUB_BUFFER_OFFSET',
    'CL_NAME_VERSION_MAX_NAME_SIZE', 'CL_NONE', 'CL_NON_BLOCKING',
    'CL_OUT_OF_HOST_MEMORY', 'CL_OUT_OF_RESOURCES',
    'CL_PIPE_MAX_PACKETS', 'CL_PIPE_PACKET_SIZE',
    'CL_PIPE_PROPERTIES', 'CL_PLATFORM_EXTENSIONS',
    'CL_PLATFORM_EXTENSIONS_WITH_VERSION',
    'CL_PLATFORM_HOST_TIMER_RESOLUTION', 'CL_PLATFORM_NAME',
    'CL_PLATFORM_NUMERIC_VERSION', 'CL_PLATFORM_PROFILE',
    'CL_PLATFORM_VENDOR', 'CL_PLATFORM_VERSION',
    'CL_PROFILING_COMMAND_COMPLETE', 'CL_PROFILING_COMMAND_END',
    'CL_PROFILING_COMMAND_QUEUED', 'CL_PROFILING_COMMAND_START',
    'CL_PROFILING_COMMAND_SUBMIT', 'CL_PROFILING_INFO_NOT_AVAILABLE',
    'CL_PROGRAM_BINARIES', 'CL_PROGRAM_BINARY_SIZES',
    'CL_PROGRAM_BINARY_TYPE',
    'CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT',
    'CL_PROGRAM_BINARY_TYPE_EXECUTABLE',
    'CL_PROGRAM_BINARY_TYPE_LIBRARY', 'CL_PROGRAM_BINARY_TYPE_NONE',
    'CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE',
    'CL_PROGRAM_BUILD_LOG', 'CL_PROGRAM_BUILD_OPTIONS',
    'CL_PROGRAM_BUILD_STATUS', 'CL_PROGRAM_CONTEXT',
    'CL_PROGRAM_DEVICES', 'CL_PROGRAM_IL', 'CL_PROGRAM_KERNEL_NAMES',
    'CL_PROGRAM_NUM_DEVICES', 'CL_PROGRAM_NUM_KERNELS',
    'CL_PROGRAM_REFERENCE_COUNT',
    'CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT',
    'CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT', 'CL_PROGRAM_SOURCE',
    'CL_QUEUED', 'CL_QUEUE_CONTEXT', 'CL_QUEUE_DEVICE',
    'CL_QUEUE_DEVICE_DEFAULT', 'CL_QUEUE_ON_DEVICE',
    'CL_QUEUE_ON_DEVICE_DEFAULT',
    'CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE',
    'CL_QUEUE_PROFILING_ENABLE', 'CL_QUEUE_PROPERTIES',
    'CL_QUEUE_PROPERTIES_ARRAY', 'CL_QUEUE_REFERENCE_COUNT',
    'CL_QUEUE_SIZE', 'CL_R', 'CL_RA', 'CL_READ_ONLY_CACHE',
    'CL_READ_WRITE_CACHE', 'CL_RG', 'CL_RGB', 'CL_RGBA', 'CL_RGBx',
    'CL_RGx', 'CL_RUNNING', 'CL_Rx', 'CL_SAMPLER_ADDRESSING_MODE',
    'CL_SAMPLER_CONTEXT', 'CL_SAMPLER_FILTER_MODE',
    'CL_SAMPLER_LOD_MAX', 'CL_SAMPLER_LOD_MIN',
    'CL_SAMPLER_MIP_FILTER_MODE', 'CL_SAMPLER_NORMALIZED_COORDS',
    'CL_SAMPLER_PROPERTIES', 'CL_SAMPLER_REFERENCE_COUNT',
    'CL_SIGNED_INT16', 'CL_SIGNED_INT32', 'CL_SIGNED_INT8',
    'CL_SNORM_INT16', 'CL_SNORM_INT8', 'CL_SUBMITTED', 'CL_SUCCESS',
    'CL_TRUE', 'CL_UNORM_INT16', 'CL_UNORM_INT24', 'CL_UNORM_INT8',
    'CL_UNORM_INT_101010', 'CL_UNORM_INT_101010_2',
    'CL_UNORM_SHORT_555', 'CL_UNORM_SHORT_565', 'CL_UNSIGNED_INT16',
    'CL_UNSIGNED_INT32', 'CL_UNSIGNED_INT8', 'CL_VERSION_MAJOR_BITS',
    'CL_VERSION_MAJOR_MASK', 'CL_VERSION_MINOR_BITS',
    'CL_VERSION_MINOR_MASK', 'CL_VERSION_PATCH_BITS',
    'CL_VERSION_PATCH_MASK', 'CL_sBGRA', 'CL_sRGB', 'CL_sRGBA',
    'CL_sRGBx', '__OPENCL_CL_H', 'clBuildProgram', 'clCloneKernel',
    'clCompileProgram', 'clCreateBuffer',
    'clCreateBufferWithProperties', 'clCreateCommandQueue',
    'clCreateCommandQueueWithProperties', 'clCreateContext',
    'clCreateContextFromType', 'clCreateImage', 'clCreateImage2D',
    'clCreateImage3D', 'clCreateImageWithProperties',
    'clCreateKernel', 'clCreateKernelsInProgram', 'clCreatePipe',
    'clCreateProgramWithBinary', 'clCreateProgramWithBuiltInKernels',
    'clCreateProgramWithIL', 'clCreateProgramWithSource',
    'clCreateSampler', 'clCreateSamplerWithProperties',
    'clCreateSubBuffer', 'clCreateSubDevices', 'clCreateUserEvent',
    'clEnqueueBarrier', 'clEnqueueBarrierWithWaitList',
    'clEnqueueCopyBuffer', 'clEnqueueCopyBufferRect',
    'clEnqueueCopyBufferToImage', 'clEnqueueCopyImage',
    'clEnqueueCopyImageToBuffer', 'clEnqueueFillBuffer',
    'clEnqueueFillImage', 'clEnqueueMapBuffer', 'clEnqueueMapImage',
    'clEnqueueMarker', 'clEnqueueMarkerWithWaitList',
    'clEnqueueMigrateMemObjects', 'clEnqueueNDRangeKernel',
    'clEnqueueNativeKernel', 'clEnqueueReadBuffer',
    'clEnqueueReadBufferRect', 'clEnqueueReadImage',
    'clEnqueueSVMFree', 'clEnqueueSVMMap', 'clEnqueueSVMMemFill',
    'clEnqueueSVMMemcpy', 'clEnqueueSVMMigrateMem',
    'clEnqueueSVMUnmap', 'clEnqueueTask', 'clEnqueueUnmapMemObject',
    'clEnqueueWaitForEvents', 'clEnqueueWriteBuffer',
    'clEnqueueWriteBufferRect', 'clEnqueueWriteImage', 'clFinish',
    'clFlush', 'clGetCommandQueueInfo', 'clGetContextInfo',
    'clGetDeviceAndHostTimer', 'clGetDeviceIDs', 'clGetDeviceInfo',
    'clGetEventInfo', 'clGetEventProfilingInfo',
    'clGetExtensionFunctionAddress',
    'clGetExtensionFunctionAddressForPlatform', 'clGetHostTimer',
    'clGetImageInfo', 'clGetKernelArgInfo', 'clGetKernelInfo',
    'clGetKernelSubGroupInfo', 'clGetKernelWorkGroupInfo',
    'clGetMemObjectInfo', 'clGetPipeInfo', 'clGetPlatformIDs',
    'clGetPlatformInfo', 'clGetProgramBuildInfo', 'clGetProgramInfo',
    'clGetSamplerInfo', 'clGetSupportedImageFormats', 'clLinkProgram',
    'clReleaseCommandQueue', 'clReleaseContext', 'clReleaseDevice',
    'clReleaseEvent', 'clReleaseKernel', 'clReleaseMemObject',
    'clReleaseProgram', 'clReleaseSampler', 'clRetainCommandQueue',
    'clRetainContext', 'clRetainDevice', 'clRetainEvent',
    'clRetainKernel', 'clRetainMemObject', 'clRetainProgram',
    'clRetainSampler', 'clSVMAlloc', 'clSVMFree',
    'clSetContextDestructorCallback',
    'clSetDefaultDeviceCommandQueue', 'clSetEventCallback',
    'clSetKernelArg', 'clSetKernelArgSVMPointer',
    'clSetKernelExecInfo', 'clSetMemObjectDestructorCallback',
    'clSetProgramReleaseCallback',
    'clSetProgramSpecializationConstant', 'clSetUserEventStatus',
    'clUnloadCompiler', 'clUnloadPlatformCompiler', 'clWaitForEvents',
    'cl_addressing_mode', 'cl_bitfield', 'cl_bool',
    'cl_buffer_create_type', 'cl_buffer_region', 'cl_build_status',
    'cl_channel_order', 'cl_channel_type', 'cl_command_queue',
    'cl_command_queue_info', 'cl_command_queue_properties',
    'cl_command_type', 'cl_context', 'cl_context_info',
    'cl_context_properties', 'cl_device_affinity_domain',
    'cl_device_atomic_capabilities',
    'cl_device_device_enqueue_capabilities',
    'cl_device_exec_capabilities', 'cl_device_fp_config',
    'cl_device_id', 'cl_device_info', 'cl_device_local_mem_type',
    'cl_device_mem_cache_type', 'cl_device_partition_property',
    'cl_device_svm_capabilities', 'cl_device_type', 'cl_event',
    'cl_event_info', 'cl_filter_mode', 'cl_image_desc',
    'cl_image_format', 'cl_image_info', 'cl_int', 'cl_kernel',
    'cl_kernel_arg_access_qualifier',
    'cl_kernel_arg_address_qualifier', 'cl_kernel_arg_info',
    'cl_kernel_arg_type_qualifier', 'cl_kernel_exec_info',
    'cl_kernel_info', 'cl_kernel_sub_group_info',
    'cl_kernel_work_group_info', 'cl_khronos_vendor_id',
    'cl_map_flags', 'cl_mem', 'cl_mem_flags', 'cl_mem_info',
    'cl_mem_migration_flags', 'cl_mem_object_type',
    'cl_mem_properties', 'cl_name_version', 'cl_pipe_info',
    'cl_pipe_properties', 'cl_platform_id', 'cl_platform_info',
    'cl_profiling_info', 'cl_program', 'cl_program_binary_type',
    'cl_program_build_info', 'cl_program_info', 'cl_properties',
    'cl_queue_properties', 'cl_sampler', 'cl_sampler_info',
    'cl_sampler_properties', 'cl_svm_mem_flags', 'cl_uint',
    'cl_version', 'size_t', 'struct__cl_buffer_region',
    'struct__cl_command_queue', 'struct__cl_context',
    'struct__cl_device_id', 'struct__cl_event',
    'struct__cl_image_desc', 'struct__cl_image_format',
    'struct__cl_kernel', 'struct__cl_mem', 'struct__cl_name_version',
    'struct__cl_platform_id', 'struct__cl_program',
    'struct__cl_sampler', 'union__cl_image_desc_0']


# tinygrad/