import ctypes
from ctypes import byref
import typing
import warnings
import functools
import itertools
import numpy as np
# import const
import sqlheaders as headers

from importlib import reload
reload(headers)

odbc = ctypes.windll.odbc32

def npref(np_obj, offset=0):
    return ctypes.c_void_p(np_obj.ctypes.data + offset)

sql_c_to_numpy_dtypes = {}

for dt in [
 'datetimeoffset', 'time', 'hierarchyid', 'sql_variant',
 'uniqueidentifier', 'bit', 'tinyint', 'bigint',
 'numeric', 'decimal', 'int',
 'smallint', 'float', 'real', 'date', 'datetime'
]:
    sql_c_to_numpy_dtypes[(dt, "SQL_C_WCHAR")] = "|V200"
    sql_c_to_numpy_dtypes[(dt, "SQL_C_CHAR")] = "|V100"

sql_c_to_numpy_dtypes.update({
    ("numeric", "SQL_C_CHAR"): (numeric_char := lambda sql_size: f"S{2 + int(np.ceil(sql_size * np.emath.logn(8, 10)))}"),
    ("decimal", "SQL_C_CHAR"): numeric_char,
    ("nchar", "SQL_C_WCHAR"): (nchar := lambda sql_size: f"V{2*(sql_size + 1)}"), # null terminated
    ("nvarchar", "SQL_C_WCHAR"): nchar,
    ("char", "SQL_C_CHAR"): (char := lambda sql_size: f"S{sql_size + 1}"),
    ("varchar", "SQL_C_CHAR"): char,
    ('bit', "SQL_C_BIT"): "bool",
    ('tinyint', "SQL_C_UTINYINT"): "uint8",
    ('smallint', "SQL_C_SHORT"): "int16",
    ('int', "SQL_C_SLONG"): "int32",
    ('bigint', "SQL_C_SBIGINT"): "int64",
    # ('date', "SQL_C_TYPE_TIMESTAMP"): "S24",
    # ('date', "SQL_C_TIME"): "S24",
    ('date', "SQL_C_TYPE_DATE"): "S6",
    ('datetime', "SQL_C_TYPE_TIMESTAMP"): "S16",
    # ('date', "SQL_C_CHAR"): "S100",
    # ('datetime', "SQL_C_CHAR"): "S200",
    
    ("char", "SQL_C_WCHAR"): nchar,
    ("varchar", "SQL_C_WCHAR"): nchar,
    ("binary", "SQL_C_WCHAR"): (binary := lambda sql_size: f"V{4*sql_size}"),
    ("varbinary", "SQL_C_WCHAR"): binary,
})

# default is based on dict order
sql_c_type_map_default = {k: v for k, v in sql_c_to_numpy_dtypes.keys()}
sql_c_type_map_wchar = {k: v for k, v in sql_c_to_numpy_dtypes.keys() if v == "SQL_C_WCHAR"}

sql_size_max = 8192
partial_sql_size = 2048

def c_metadata_from_sql(metadata, sql_c_type_map=None):
    """
    Add metadata for numpy datastructures and SQL_C_TYPES based on sql metadata.
    Bound columns must be at the start of the column list, see:
    https://learn.microsoft.com/en-us/sql/odbc/reference/develop-app/getting-long-data
    As a consequence, all columns following the first non bindable column will
    be marked non bindable.
    """
    if sql_c_type_map is None:
        sql_c_type_map = sql_c_type_map_default
    previous_bindable = True
    for meta_col in metadata:
        sql_type_name = headers.sql_types[meta_col["sql_type"]]
        c_type_name = sql_c_type_map[sql_type_name]
        meta_col["sql_type_name"] = sql_type_name
        meta_col["c_type_name"] = c_type_name
        meta_col["c_type"] = headers.c_types[c_type_name]
        dt = sql_c_to_numpy_dtypes[(sql_type_name, c_type_name)]
        meta_col["partial_fetch"] = bool(
            meta_col["sql_size"] <= 0
            or meta_col["sql_size"] > sql_size_max
        )
        meta_col["bindable"] = previous_bindable = bool(
            not meta_col["partial_fetch"] and previous_bindable
        )
        meta_col["np_dtype_name"] = dt if type(dt) is str else dt(
            partial_sql_size if meta_col["partial_fetch"] else meta_col["sql_size"]
        )

def length_from_max_bytes(metadata, max_bytes, max_length):
    itemsize = sum(
        np.dtype(meta_col["np_dtype_name"]).itemsize
        for meta_col in metadata
    )
    return min(max_length, max_bytes//itemsize)

def create_buffers(metadata, length):
    """
    Create for a result set based on metadata. The buffers will have space for
    `length` rows. The needed metadata consists of only `np_dtype_name`.
    """
    values = []
    for meta_col in metadata:
        buffer = np.zeros(length, dtype=np.dtype(meta_col["np_dtype_name"]))
        actual_width = np.zeros(length, dtype=np.int64)
        values.append((buffer, actual_width))
    return values

def create_buffers_row(metadata, length):
    """
    Create buffer for a result set based on metadata and update metadata with
    offsets. The buffer will have space for `length` rows. The needed metadata
    consists of only `np_dtype_name` and `offset_data` as well as `offset_len`
    will be added to metadata.
    """
    values = []
    itemsize = 0
    for meta_col in metadata:
        col_size = np.dtype(meta_col["np_dtype_name"]).itemsize
        meta_col["offset_data"] = itemsize
        meta_col["offset_len"] = itemsize + col_size
        itemsize += col_size + 8
    values = np.zeros(length, dtype=f"S{itemsize}")
    return values


class NoData(Exception):
    pass

# TODO: properly insert into __doc__ (.replace), figure out transition from header string to number
def _set_options(**options):
    def inner(func):
        func.options = options
        func.__doc__ += "\n" + repr(options)
        return func
    return inner


class _base:
    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def alloc_handle(self):
        self.check_error(odbc.SQLAllocHandle(
            getattr(headers, self.handle_type),
            self.parent.handle if hasattr(self, "parent") else headers.SQL_NULL_HANDLE,
            byref(handle := ctypes.c_void_p())
        ))
        self.handle = handle

    def close(self) -> None:
        if hasattr(self, "children"):
            for child in self.children:
                child.close()
        self.check_error(odbc.SQLFreeHandle(self.handle_type, self.handle), warn_only=True)
    
    def check_error(self, ret, warn_only=False):
        # by default python treats return values of C functions as unsigned short.
        # or does it?
        if ret > 2**15:
            ret = ret - 2**16
        if ret == headers.SQL_SUCCESS:
            return
        if ret == headers.SQL_NO_DATA:
            raise NoData(f"ret = SQL_NO_DATA = {headers.SQL_NO_DATA}")
        ret2 = odbc.SQLGetDiagRecW(
            getattr(headers, self.handle_type),
            self.handle,
            1,
            state := ctypes.create_unicode_buffer(24),
            byref(native := ctypes.c_int()),
            msg := ctypes.create_unicode_buffer(1024*16),
            len(msg),
            byref(buf_len := ctypes.c_short())
        )
        if ret2 == headers.SQL_NO_DATA:
            return
        if ret2 != headers.SQL_SUCCESS:
            print("uh oh", ret2)
        err = ret, native.value, state.value, msg.value
        if ret == headers.SQL_SUCCESS_WITH_INFO or warn_only:
            # warnings.warn(repr(err), stacklevel=2)
            warnings.warn(repr(err), stacklevel=3)
            # warnings.warn(repr(err), stacklevel=4)
        else:
            raise Exception(err)




def _stmt_attr_property(attr, doc=None):
    return property(
        lambda self: self.SQLGetStmtAttrInt(attr),
        lambda self, val: self.SQLSetStmtAttrInt(attr, val),
        doc=doc
    )

class stmt(_base):
    def __init__(self, dbc) -> None:
        self.handle_type = "SQL_HANDLE_STMT"
        self.parent = dbc
        self.parent.children.append(self)
        self.alloc_handle()

    def close(self):
        for opt in [
            "SQL_RESET_PARAMS",
            "SQL_UNBIND",
            "SQL_CLOSE"
        ]:
            self.SQLFreeStmt(getattr(headers, opt))
        super().close()

    def SQLFreeStmt(self, option):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlfreestmt-function"
        self.check_error(odbc.SQLFreeStmt(
            self.handle,
            option
        ), warn_only=True)

    def SQLExecDirect(self, sql):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlexecdirect-function"
        try:
            self.check_error(odbc.SQLExecDirectW(
                self.handle,
                ctypes.c_wchar_p(sql),
                len(sql)
            ))
        except NoData: # Databricks does this, not sure why
            pass
    
    def SQLPrepare(self, sql):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlprepare-function"
        self.check_error(odbc.SQLPrepareW(
            self.handle,
            ctypes.c_wchar_p(sql),
            len(sql)
        ))

    def SQLExecute(self):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlexecute-function"
        try:
            self.check_error(odbc.SQLExecute(self.handle))
        except NoData: # Databricks does this, not sure why
            pass
    
    def SQLSetStmtAttr(self, attr, val, str_len):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlsetstmtattr-function"
        self.check_error(odbc.SQLSetStmtAttr(  
            self.handle,
            attr, val, str_len
        ))

    def SQLGetStmtAttr(self, attr):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlgetstmtattr-function"
        self.check_error(odbc.SQLGetStmtAttr(  
            self.handle,
            attr, buf := ctypes.create_string_buffer(4 * 1024), len(buf), None
        ))
        return buf.raw

    def SQLSetStmtAttrInt(self, attr, val):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlsetstmtattr-function"
        self.check_error(odbc.SQLSetStmtAttr(  
            self.handle,
            attr, val, headers.SQL_IS_INTEGER
        ))

    def SQLGetStmtAttrInt(self, attr):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlgetstmtattr-function"
        self.check_error(odbc.SQLGetStmtAttr(  
            self.handle,
            attr, buf := ctypes.create_string_buffer(8), len(buf), None
        ))
        return int.from_bytes(buf.raw, "little")

    def SQLDescribeParam(self, param):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqldescribeparam-function"
        self.check_error(odbc.SQLDescribeParam(  
            self.handle, param,
            ctypes.byref(DataType := ctypes.c_short()),
            ctypes.byref(ParamSize := ctypes.c_ulonglong()),
            ctypes.byref(DecimalDigits := ctypes.c_short()),
            ctypes.byref(Nullable := ctypes.c_short()),
        ))
        return DataType.value, ParamSize.value, DecimalDigits.value, Nullable.value

    def SQLDescribeCol(self, col):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqldescribecol-function"
        self.check_error(odbc.SQLDescribeColW(
            self.handle,
            col,
            ColumnName := ctypes.create_unicode_buffer(1024),
            len(ColumnName),
            byref(NameLength := ctypes.c_short()),
            byref(DataType := ctypes.c_short()),
            byref(ColumnSize := ctypes.c_size_t()),
            byref(DecimalDigits := ctypes.c_short()),
            byref(Nullable := ctypes.c_short())
        ))
        return ColumnName.value, NameLength.value, DataType.value, ColumnSize.value, DecimalDigits.value, Nullable.value

    @_set_options(
        io_type=(
            "SQL_PARAM_INPUT", "SQL_PARAM_INPUT_OUTPUT", "SQL_PARAM_OUTPUT",
            "SQL_PARAM_INPUT_OUTPUT_STREAM", "SQL_PARAM_OUTPUT_STREAM"
        ),
        StrLen_or_IndPtr=(
            "SQL_NTS", "SQL_NULL_DATA", "SQL_DEFAULT_PARAM", "SQL_DATA_AT_EXEC",
            None # null ptr
        )
    )
    def SQLBindParameter(
            self, param, io_type, c_type, sql_type, sql_size, dec_digits,
            param_ref, buffer_width, StrLen_or_IndPtr
        ):
        """https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlbindparameter-function
        sql_size, dec_digits describes the size of the sql type
        buffer_width describes the size (number of bytes) of *one* row in the parameter buffer array
        StrLen_or_IndPtr has a row for every row in the parameter buffer describing the length of the c type. -1 indicates null
        """
        self.check_error(odbc.SQLBindParameter(  
            self.handle, param,
            io_type,
            c_type,
            sql_type,
            sql_size,
            dec_digits,
            param_ref,
            buffer_width,
            StrLen_or_IndPtr,
        ))

    def SQLNumParams(self):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlnumparams-function"
        self.check_error(odbc.SQLNumParams(
            self.handle,
            byref(num_cols := ctypes.c_int16())
        ))
        return num_cols.value
    
    def SQLNumResultCols(self):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlnumresultcols-function"
        self.check_error(odbc.SQLNumResultCols(
            self.handle,
            byref(num_cols := ctypes.c_int16())
        ))
        return num_cols.value

    def SQLColAttribute(self, col):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlcolattribute-function"
        self.check_error(odbc.SQLColAttribute(
            self.handle,
            col,
            p.SQL_COLUMN_DISPLAY_SIZE,
            byref(CharacterAttribute := ctypes.create_string_buffer(10)),
            len(CharacterAttribute),
            byref(StringLength := ctypes.c_int16()),
            byref(NumericAttribute := ctypes.c_int64()),
        ))
        return CharacterAttribute.value, StringLength.value, NumericAttribute.value

    def SQLBindCol(
            self, column, c_type, buffer_ref, buffer_width, StrLen_or_IndPtr
        ):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlbindcol-function"
        self.check_error(odbc.SQLBindCol(  
            self.handle,
            column,
            c_type,
            buffer_ref,
            buffer_width,
            StrLen_or_IndPtr
        ))

    def SQLGetData(self, index, c_type, buffer_ref, buffer_width, StrLen_or_IndPtr):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlgetdata-function"
        self.check_error(odbc.SQLGetData(  
            self.handle, index, c_type, buffer_ref, buffer_width, StrLen_or_IndPtr
        ))

    def SQLPutData(self, buffer_ref, StrLen_or_IndPtr):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlputdata-function"
        self.check_error(odbc.SQLFetch(  
            self.handle, buffer_ref, StrLen_or_IndPtr
        ))

    def SQLFetch(self):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlfetch-function"
        self.check_error(odbc.SQLFetch(  
            self.handle,
        ))

    @_set_options(
        operation = ["SQL_ADD", "SQL_UPDATE_BY_BOOKMARK",
                     "SQL_DELETE_BY_BOOKMARK", "SQL_FETCH_BY_BOOKMARK"]
    )
    def SQLBulkOperations(self, operation):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlbulkoperations-function"
        self.check_error(odbc.SQLBulkOperations(  
            self.handle,
            operation
        ))

    @_set_options(
        operation = ["SQL_POSITION", "SQL_REFRESH", "SQL_UPDATE", "SQL_DELETE"],
        lock = ["SQL_LOCK_NO_CHANGE", "SQL_LOCK_EXCLUSIVE", "SQL_LOCK_UNLOCK"]
    )
    def SQLSetPos(self, row, operation, lock):
        """https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlsetpos-function
        May require setting self.cursor_type to a value other than SQL_CURSOR_FORWARD_ONLY"""
        self.check_error(odbc.SQLSetPos(  
            self.handle, row, operation, lock
        ))

    def SQLFetchScroll(self, orientation, offset):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlfetchscroll-function"
        self.check_error(odbc.SQLFetchScroll(  
            self.handle, orientation, offset
        ))

    def SQLRowCount(self):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlrowcount-function"
        self.check_error(odbc.SQLRowCount(  
            self.handle, byref(rowcount := headers.SQLLEN())
        ))
        return rowcount.value
        

    # TODO SetDescField for schema changes
    # https://learn.microsoft.com/en-us/sql/relational-databases/native-client-odbc-table-valued-parameters/uses-of-odbc-table-valued-parameters?view=sql-server-ver16#table-valued-parameter-with-fully-bound-multirow-buffers-send-data-as-a-tvp-with-all-values-in-memory
    # TODO SQLColAttribute to automatically get string buffer sizes
    # https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlcolattribute-function

    def sql_metadata(self, kind: typing.Literal["result", "parameter"] = "result"):
        "obtain metadata using SQLDescribeCol or SQLDescribeParam"
        # TODO potentially autodetect if query has been executed and set `kind` accordingly
        metadata = []
        isres = {"result": True, "parameter": False}[kind]
        for col in range(1, (self.SQLNumResultCols() if isres else self.SQLNumParams())+1):
            meta_col = {}
            metadata.append(meta_col)
            if isres:
                (
                    meta_col["name"], _, meta_col["sql_type"], meta_col["sql_size"],
                    meta_col["dec_digits"], meta_col["nullable"]
                ) = self.SQLDescribeCol(col)
            else:
                (
                    meta_col["sql_type"], meta_col["sql_size"],
                    meta_col["dec_digits"], meta_col["nullable"]
                ) = self.SQLDescribeParam(col)
        return metadata

    def bind_columns(self, metadata, buffers):
        "Bind buffers column-wise to columns of result set"
        for col, (meta_col, (buffer, actual_width)) in enumerate(zip(metadata, buffers, strict=True), 1):
            if not meta_col["bindable"]:
                continue
            self.SQLBindCol(
                col, meta_col["c_type"],
                npref(buffer), buffer.itemsize, npref(actual_width)
            )

    def bind_columns_late(self, metadata, buffers, rowcount):
        "Use SQLGetData for unbound columns after fetching"
        extra_data = {}
        for col, (meta_col, (buffer, actual_width)) in enumerate(zip(metadata, buffers, strict=True), 1):
            if meta_col["bindable"]:
                continue
            extra_buffer = np.zeros(1, dtype=buffer.dtype)
            extra_width = np.zeros(1, dtype=actual_width.dtype)
            assert rowcount <= len(actual_width)
            for pos in range(rowcount):
                if len(actual_width) > 1:
                    self.SQLSetPos(
                        pos+1,
                        headers.SQL_POS_POSITION,
                        headers.SQL_LOCK_NO_CHANGE
                    )
                print(col, pos)
                self.SQLGetData(
                    col,
                    meta_col["c_type"],
                    npref(buffer, offset=pos*buffer.itemsize),
                    buffer.itemsize,
                    npref(actual_width, offset=pos*actual_width.itemsize)
                )
                if actual_width[pos] == headers.SQL_NULL_DATA:
                    continue # GetData never raises NoData -> loop
                while True:
                    try:
                        self.SQLGetData(
                            col,
                            meta_col["c_type"],
                            npref(extra_buffer),
                            extra_buffer.itemsize,
                            npref(extra_width)
                        )
                    except NoData:
                        break
                    else:
                        # cp = extra_buffer[0], extra_width[0] # this copies
                        # cp = extra_buffer.copy(), extra_width.copy()
                        cp = extra_buffer.copy(), extra_width[0]
                        if col in extra_data:
                            if pos in extra_data[col]:
                                extra_data[col][pos].append(cp)
                            else:
                                extra_data[col][pos] = [cp]
                        else:
                            extra_data[col] = {pos: [cp]}
        return extra_data

    def bind_columns_row(self, metadata, buffer):
        "Bind buffer row-wise to columns of result set"
        for col, meta_col in enumerate(metadata, 1):
            self.SQLBindCol(
                col, meta_col["c_type"],
                npref(buffer, offset=meta_col["offset_data"]),
                np.dtype(meta_col["np_dtype_name"]).itemsize,
                npref(buffer, offset=meta_col["offset_len"])
            )

    def bind_params(self, metadata, values):
        "bind `values` as parameters using `metadata`"
        for col, (meta_col, val) in enumerate(zip(metadata, values, strict=True), 1):
            if meta_col["sql_type"] == headers.sql_types["SQL_SS_TABLE"]:
                length = meta_col["length"]
                self.SQLBindParameter(
                    col, headers.SQL_PARAM_INPUT,
                    headers.SQL_C_DEFAULT, meta_col["sql_type"],
                    length, 0,
                    ctypes.c_wchar_p(meta_col["name"]), headers.SQL_NTS,
                    npref(np.array([length], dtype=np.int64))
                )
                self.parameter_focus = col
                self.bind_params(meta_col["metadata"], val)
                self.parameter_focus = 0
            else:
                (buffer, actual_width) = val
                self.SQLBindParameter(
                    col, headers.SQL_PARAM_INPUT,
                    meta_col["c_type"], meta_col["sql_type"], 
                    meta_col["sql_size"], meta_col["dec_digits"],
                    npref(buffer), buffer.itemsize, npref(actual_width)
                )

    def bind_params_row(self, metadata, values):
        "bind `values` as parameters using `metadata`"
        for col, meta_col in enumerate(metadata, 1):
            if meta_col["sql_type"] == headers.sql_types["SQL_SS_TABLE"]:
                raise NotImplementedError("idk")
            else:
                self.SQLBindParameter(
                    col, headers.SQL_PARAM_INPUT,
                    meta_col["c_type"], meta_col["sql_type"], 
                    meta_col["sql_size"], meta_col["dec_digits"],
                    npref(values, offset=meta_col["offset_data"]),
                    np.dtype(meta_col["np_dtype_name"]).itemsize,
                    npref(values, offset=meta_col["offset_len"])
                )

    def create_row_status(self, buffer_length):
        """
        Possible values are:
            
        SQL_ROW_SUCCESS = 0
        SQL_ROW_DELETED = 1
        SQL_ROW_UPDATED = 2
        SQL_ROW_NOROW = 3
        SQL_ROW_ADDED = 4
        SQL_ROW_ERROR = 5
        SQL_ROW_SUCCESS_WITH_INFO = 6
        """
        self.SQLSetStmtAttr(
            headers.SQL_ATTR_ROW_STATUS_PTR,
            npref(row_status := np.zeros(buffer_length, dtype=np.int16)),
            0
        )
        return row_status

    row_width = _stmt_attr_property(
        headers.SQL_ATTR_ROW_BIND_TYPE,
        doc="Byte-width of row-wise result set buffer. Is `0` for column-wise binding"
    )
    parameter_array_length = _stmt_attr_property(headers.SQL_ATTR_PARAMSET_SIZE)
    fetch_array_length = _stmt_attr_property(
        headers.SQL_ATTR_ROW_ARRAY_SIZE,
        doc="""
        Careful when using values > 1 and variable length columns:
        https://learn.microsoft.com/en-us/sql/odbc/reference/develop-app/sqlgetdata-and-block-cursors
        """
    )

    # TODO fix format: 0 = 63*2**32 apparently SQLULEN is required or something
    parameter_focus = _stmt_attr_property(headers.SQL_SOPT_SS_PARAM_FOCUS)
    concurrency_setting = _stmt_attr_property(headers.SQL_ATTR_CONCURRENCY)
    cursor_type = _stmt_attr_property(headers.SQL_ATTR_CURSOR_TYPE)


def _dbc_attr_property(attr, doc=None):
    def getter(self):
        self.check_error(odbc.SQLGetConnectAttr(
            self.handle, attr, byref(buf := headers.SQLULEN()), 8, None
        ))
        return buf.value
    def setter(self, val):
        self.check_error(odbc.SQLSetConnectAttr(
            self.handle, attr, int(val), headers.SQL_IS_UINTEGER
        ))
    return property(getter, setter, doc=doc)

class dbc(_base):
    def __init__(self, env, connect_string) -> None:
        self.handle_type = "SQL_HANDLE_DBC"
        self.parent = env
        self.parent.children.append(self)
        self.children = []
        self.alloc_handle()
        self.check_error(odbc.SQLDriverConnectW(
            self.handle, 0,
            # ctypes.create_unicode_buffer(connect_string, l := len(connect_string)),
            ctypes.c_wchar_p(connect_string),
            len(connect_string),
            out := ctypes.create_unicode_buffer(l_buf := 8*1024),
            l_buf,
            l_out := ctypes.c_longlong(),
            headers.SQL_DRIVER_NOPROMPT
        ))
        print(l_out, out.value)
        
    @functools.wraps(stmt)
    def stmt(self, *args, **kwargs) -> stmt:
        return stmt(self, *args, **kwargs)

    # TODO: seems broken
    def SQLNativeSql(self, sql):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlnativesql-function"
        self.check_error(odbc.SQLNativeSql(
            self.handle,
            ctypes.c_wchar_p(sql),
            len(sql),
            out := ctypes.create_unicode_buffer(l_buf := 8*1024),
            l_buf,
            l_out := ctypes.c_short(),
        ))
        return out, l_out
        # return out.value[:l_out.value]

    # class test: pass
    # self = test()
    # self.__class__ = dbc
    # self.handle_type = "SQL_HANDLE_DBC"
    # self.parent = e
    # self.children = []
    # self.alloc_handle()
    # bytearray(self.SQLBrowseConnect("""
    # Driver={SQL Server};
    # """)[0]).strip(b"\x00")
    def SQLBrowseConnect(self, connection_string):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqlbrowseconnect-function"
        self.check_error(odbc.SQLBrowseConnect(
            self.handle,
            ctypes.c_wchar_p(connection_string),
            len(connection_string),
            out := ctypes.create_unicode_buffer(l_buf := 8*1024),
            l_buf,
            l_out := ctypes.c_short(),
        ), warn_only=True)
        return out, l_out
        # return out.value[:l_out.value]

    def commit(self):
        self.check_error(odbc.SQLEndTran(
            getattr(headers, self.handle_type),
            self.handle,
            headers.SQL_COMMIT)
        )

    def rollback(self):
        self.check_error(odbc.SQLEndTran(
            getattr(headers, self.handle_type),
            self.handle,
            headers.SQL_ROLLBACK)
        )
    # TODO autocommit, SQLGetConnectAttr 

    autocommit = _dbc_attr_property(headers.SQL_ATTR_AUTOCOMMIT)


class env(_base):
    def __init__(self) -> None:
        self.handle_type = "SQL_HANDLE_ENV"
        self.children = []
        self.alloc_handle()
        self.check_error(odbc.SQLSetEnvAttr(
            self.handle, headers.SQL_ATTR_ODBC_VERSION, headers.SQL_OV_ODBC3, 0
        ))
    
    @functools.wraps(dbc)
    def dbc(self, *args, **kwargs) -> dbc:
        return dbc(self, *args, **kwargs)

    @_set_options(direction = ("SQL_FETCH_FIRST", "SQL_FETCH_NEXT"))
    def SQLDrivers(self, direction):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqldrivers-function"
        self.check_error(odbc.SQLDrivers(
            self.handle, direction,
            description := ctypes.create_string_buffer(l_buf_desc := 8*1024),
            l_buf_desc, l_actual_desc := ctypes.c_short(),
            attributes := ctypes.create_string_buffer(l_buf_attr := 8*1024),
            l_buf_attr, l_actual_attr := ctypes.c_short(),
        ))
        return description, attributes # maybe return something nicer

    def drivers(self):
        """Provide all drivers and attributes as a dict"""
        ret = {}
        for it in itertools.count():
            try:
                desc, attr = self.SQLDrivers(
                    headers.SQL_FETCH_FIRST if it == 0 else headers.SQL_FETCH_NEXT
                )
            except NoData:
                break
            ret.__setitem__(
                bytearray(desc).strip(b"\x00").decode(),
                bytearray(attr).strip(b"\x00").decode().split("\x00")
            )
        return ret

    # TODO: refactor copy paste
    @_set_options(direction = ("SQL_FETCH_FIRST", "SQL_FETCH_NEXT"))
    def SQLDataSources(self, direction):
        "https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/sqldatasources-function"
        self.check_error(odbc.SQLDataSources(
            self.handle, direction,
            server_name := ctypes.create_string_buffer(l_buf_sv := 8*1024),
            l_buf_sv, l_actual_sv := ctypes.c_short(),
            description := ctypes.create_string_buffer(l_buf_desc := 8*1024),
            l_buf_desc, l_actual_desc := ctypes.c_short(),
        ))
        return server_name, description # maybe return something nicer

    def data_sources(self):
        """Provide all data sources and descriptions as a dict"""
        ret = {}
        for it in itertools.count():
            try:
                desc, attr = self.SQLDataSources(
                    headers.SQL_FETCH_FIRST if it == 0 else headers.SQL_FETCH_NEXT
                )
            except NoData:
                break
            ret.__setitem__(
                bytearray(desc).strip(b"\x00").decode(),
                bytearray(attr).strip(b"\x00").decode()
            )
        return ret


def select_into(
    con_source, con_target, select_source, table_target,
    table_type_target=None, buffer_length=10000,
    mode: typing.Literal["normal", "bulk", "tvp"] = "normal",
    tablock=False, use_target_metadata=False
):
    # use_tvp = table_type_target is not None

    with (
        con_target.stmt() as stmt_target,
        con_source.stmt() as stmt_source
    ):
        row_status = stmt_source.create_row_status(buffer_length)
        if mode == "bulk":
            for stmt in (stmt_source, stmt_target):
                stmt.cursor_type = headers.SQL_CURSOR_KEYSET_DRIVEN
                stmt.concurrency_setting = headers.SQL_CONCUR_LOCK

        stmt_source.SQLExecDirect(select_source)

        metadata_source = stmt_source.sql_metadata("result")
        c_metadata_from_sql(metadata_source)
        buffers = create_buffers(metadata_source, buffer_length)

        stmt_source.bind_columns(metadata_source, buffers)
        stmt_source.fetch_array_length = buffer_length

        metadata_target = metadata_source.copy()

        match mode:
            case "tvp":
                if use_target_metadata:
                    raise NotImplementedError()
                stmt_target.bind_params(
                    [{
                        "name": table_type_target,
                        "sql_type": headers.sql_types["SQL_SS_TABLE"],
                        "metadata": metadata_target,
                        "length": buffer_length
                    }],
                    [buffers]
                )
                sql_target = (
                    f"insert into {table_target} " +
                    ("with(tablock) " if tablock else "") +
                    f"({', '.join(col['name'] for col in metadata_target)}) " +
                    "select * from ?"
                )
            case "normal":
                sql_target = (
                    f"insert into {table_target} " +
                    ("with(tablock) " if tablock else "") +
                    f"({', '.join(col['name'] for col in metadata_target)}) " +
                    f"values({', '.join('?'*len(metadata_target))})"
                )
                if use_target_metadata:
                    stmt_target.SQLPrepare(sql_target)
                    for meta_col_1, meta_col_2 in zip(
                        metadata_target, stmt_target.sql_metadata(kind="parameter"), strict=True
                    ):
                        meta_col_1.update(meta_col_2)
                stmt_target.bind_params(metadata_target, buffers)
                stmt_target.parameter_array_length = buffer_length
            case "bulk":
                if use_target_metadata:
                    raise NotImplementedError()
                stmt_target.SQLExecDirect(
                    f"select {', '.join(col['name'] for col in metadata_target)} " +
                    f"from {table_target} " +
                    ("with(tablock)" if tablock else "")
                )
                stmt_target.bind_columns(metadata_target, buffers)
                stmt_target.fetch_array_length = buffer_length
            case _:
                raise Exception(f"Invalid {mode=}")

        while True:
            try:
                stmt_source.SQLFetch()
            except NoData:
                break
            # current_length implementation assumes that once an entry equals SQL_ROW_NOROW all
            # following entrys equal the same and NoData will be raised in the next iteration
            current_length = int(
                not (ind := (row_status == headers.SQL_ROW_NOROW)).any() and buffer_length or ind.argmax()
            )
            if current_length < buffer_length:
                match mode:
                    case "tvp": raise NotImplementedError() # TODO rebind TVP column
                    case "normal": stmt_target.parameter_array_length = current_length
                    case "bulk": stmt_target.fetch_array_length = current_length
            match mode, use_target_metadata:
                case "tvp", _: stmt_target.SQLExecDirect(sql_target)
                case "normal", False: stmt_target.SQLExecDirect(sql_target)
                case "normal", True: stmt_target.SQLExecute()
                case "bulk": stmt_target.SQLBulkOperations(headers.SQL_ADD)
        

def select_into_csv(
    con, select, file_target,
    buffer_length=10000,
    sep=b'A\x94\xf1\x950\xfa',
    end=b"_\x00",
    byte_order_mark=False
):
    with (
        con.stmt() as stmt
    ):
        if buffer_length > 1:
            stmt.cursor_type = headers.SQL_CURSOR_DYNAMIC
        row_status = stmt.create_row_status(buffer_length)
        stmt.SQLExecDirect(select)
        
        rowcount = stmt.SQLRowCount()
        print(rowcount)
        metadata = stmt.sql_metadata("result")
        c_metadata_from_sql(metadata, sql_c_type_map=sql_c_type_map_wchar)
        buffers = create_buffers(metadata, buffer_length)

        stmt.bind_columns(metadata, buffers)
        stmt.fetch_array_length = buffer_length
        
        if byte_order_mark: # assumes utf-16-le, bcp does this
            file_target.write(b'\xff\xfe')
        
        while True:
            try:
                stmt.SQLFetch()
            except NoData:
                return
            # current_length implementation assumes that once an entry equals SQL_ROW_NOROW all
            # following entrys equal the same and NoData will be raised in the next iteration
            current_length = int(
                not (ind := (row_status == headers.SQL_ROW_NOROW)).any() and buffer_length or ind.argmax()
            )
            extra_data = stmt.bind_columns_late(metadata, buffers, current_length)
            def cut(buf, width):
                match width:
                    case headers.SQL_NULL_DATA:
                        w = 0
                    case headers.SQL_NO_TOTAL:
                        w = -2 # remove null terminator, different if not 16bit
                    case _ if width >= 0:
                        w = width
                        if w == 0: print(buf[:5])
                    case _:
                        raise Exception()
                return buf.view("|V1")[:w]
            def reformat(buf):
                if (l := len(buf)) == 0:
                    return b""
                else:
                    return buf.view(f"|V{l}")[0].tobytes()
            def cut_and_combined(col):
                buffer, actual_width = buffers[col]
                # -2 may be different for char/binary
                try:
                    extra = extra_data[col]
                except KeyError:
                    for pos in range(current_length):
                        yield reformat(cut(buffer[[pos]], actual_width[pos]))
                else:
                    for pos in range(current_length):
                        yield reformat(np.concatenate([
                            cut(b, w)
                            for b, w in itertools.chain(
                                ((buffer[[pos]], actual_width[pos]),),
                                extra.get(pos, ())
                            )
                        ]))
            column_generators = []
            for data in zip(
                *(cut_and_combined(col) for col in range(len(metadata))),
                itertools.cycle((end,))
            ):
                file_target.write(sep.join(data))
        
        