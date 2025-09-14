import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Literal, Optional

PeriodType = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]

class AKDataClient:
    """
    统一封装 A 股 / 港股 历史行情抓取，返回标准化 DataFrame:
    columns: [date, open, high, low, close, volume, amount, adj_factor, market, symbol]
    - 通过 data/cache 持久化存储日线数据（CSV），非日线均由日线重采样得到，减少对外请求
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _normalize_hk_symbol(symbol: str) -> str:
        s = str(symbol).upper().replace(".HK", "").strip()
        # 保留数字，若是纯数字则补零至5位
        if s.isdigit():
            return s.zfill(5)
        return s

    @staticmethod
    def _normalize_a_prefixed(symbol: str) -> str:
        """
        将输入的A股代码规范为带交易所前缀的小写格式，供 ak.stock_zh_a_daily 使用。
        规则：
        - 若已是 shXXXXXX / szXXXXXX（大小写不敏感），统一转小写返回
        - 支持形如 600000、SH600000、sz000001、000001.SZ 等格式
        - 前缀判定：以股票代码首位/前缀判断，'5','6','9','688','689' -> sh，否则 -> sz
        """
        s = str(symbol).upper().strip()
        # 去除常见分隔符
        s = s.replace(".", "").replace("-", "").replace(" ", "")
        if s.startswith("SH") or s.startswith("SZ"):
            pre = s[:2].lower()
            code = s[2:]
            # 只保留后6位数字（部分输入带市场代码或多余字符）
            digits = "".join(ch for ch in code if ch.isdigit())
            if len(digits) >= 6:
                code = digits[-6:]
            return f"{pre}{code}"
        # 形如 000001SZ / 600000SH
        if s.endswith("SH") or s.endswith("SZ"):
            code_part = s[:-2]
            exch = s[-2:].lower()
            digits = "".join(ch for ch in code_part if ch.isdigit())
            code = digits[-6:] if len(digits) >= 6 else digits.zfill(6)
            return f"{exch}{code}"
        # 纯数字或含数字优先提取后6位
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 6:
            code = digits[-6:]
        else:
            code = digits.zfill(6)
        # 简易交易所判断
        if code.startswith(("5", "6", "9")) or code.startswith("688") or code.startswith("689"):
            return f"sh{code}"
        return f"sz{code}"

    @staticmethod
    def _standardize(df: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "date","open","high","low","close","volume","amount","adj_factor","market","symbol"
            ])
        # 尝试识别常见字段名（覆盖多源差异）
        rename_map = {
            # 日期
            "日期": "date", "date": "date",
            # 开高低收
            "开盘": "open", "开盘价": "open", "open": "open",
            "最高": "high", "最高价": "high", "high": "high",
            "最低": "low", "最低价": "low", "low": "low",
            "收盘": "close", "收盘价": "close", "close": "close",
            # 成交量/额
            "成交量": "volume", "成交量(股)": "volume", "成交量(手)": "volume", "volume": "volume", "vol": "volume",
            "成交额": "amount", "成交额(元)": "amount", "成交额(港元)": "amount", "amount": "amount", "turnover": "amount",
            # 复权
            "复权因子": "adj_factor", "adj_factor": "adj_factor",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        # 转换日期
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])  # type: ignore
        elif "日期" in df.columns:
            df["date"] = pd.to_datetime(df["日期"])  # type: ignore
        # 填充缺失列
        for col in ["open","high","low","close","volume","amount","adj_factor"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["market"] = market
        df["symbol"] = symbol
        cols = ["date","open","high","low","close","volume","amount","adj_factor","market","symbol"]
        df = df[cols].sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _resample_ohlcv(df_std: pd.DataFrame, period: PeriodType) -> pd.DataFrame:
        if df_std is None or df_std.empty:
            return df_std
        rule = {
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "Y",
        }.get(period)
        if rule is None:
            return df_std
        x = df_std.set_index("date").sort_index()
        o = x["open"].resample(rule).first()
        h = x["high"].resample(rule).max()
        l = x["low"].resample(rule).min()
        c = x["close"].resample(rule).last()
        v = x["volume"].resample(rule).sum(min_count=1)
        amt = x["amount"].resample(rule).sum(min_count=1)
        # 若存在复权因子，沿用该周期内最后一个（与收盘同口径）
        if "adj_factor" in x.columns:
            try:
                af = x["adj_factor"].astype(float).resample(rule).last()
            except Exception:
                af = x["adj_factor"].resample(rule).last()
        else:
            af = None
        out = pd.concat([o,h,l,c,v,amt], axis=1)
        out.columns = ["open","high","low","close","volume","amount"]
        out = out.dropna(subset=["open","high","low","close"], how="any").reset_index()
        if af is not None:
            out = out.merge(af.reset_index().rename(columns={"adj_factor":"_af"}), on="date", how="left")
            out["adj_factor"] = out.pop("_af")
        else:
            out["adj_factor"] = pd.NA
        out["market"] = df_std["market"].iloc[0] if not df_std.empty else pd.NA
        out["symbol"] = df_std["symbol"].iloc[0] if not df_std.empty else pd.NA
        return out

    # 新增：缓存路径与读写
    def _cache_file_daily(self, market: str, symbol: str, adjust: Optional[str]) -> str:
        adj = adjust if adjust in ("qfq", "hfq") else "none"
        if market == "H":
            safe_symbol = self._normalize_hk_symbol(str(symbol))
        else:
            safe_symbol = str(symbol).replace("/", "_")
        d = os.path.join(self.cache_dir, market, safe_symbol)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"daily_{adj}.csv")

    def _is_expired(self, path: str, expire_days: int) -> bool:
        if expire_days <= 0:
            return False
        if not os.path.exists(path):
            return True
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) > timedelta(days=expire_days)

    def _load_cached_daily(self, market: str, symbol: str, adjust: Optional[str]) -> pd.DataFrame:
        path = self._cache_file_daily(market, symbol, adjust)
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])  # 确保日期类型
            return df
        except Exception:
            return pd.DataFrame()

    def _save_cached_daily(self, df_std: pd.DataFrame, market: str, symbol: str, adjust: Optional[str]) -> None:
        path = self._cache_file_daily(market, symbol, adjust)
        try:
            df_std.to_csv(path, index=False)
        except Exception:
            pass

    # 新增：从源拉取日线并标准化
    def _fetch_daily_a(self, symbol: str, adjust: Optional[str]) -> pd.DataFrame:
        # 当选择前/后复权时，同时抓取未复权数据用于计算复权因子；未复权则直接返回并将因子置为 1
        if adjust in ("qfq", "hfq"):
            try:
                df_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily")
            except Exception:
                df_raw = pd.DataFrame()
            try:
                df_adj = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust=adjust)
            except Exception:
                df_adj = pd.DataFrame()
            # 回退：若加权数据为空，尝试带交易所前缀的另一接口
            if df_adj is None or df_adj.empty:
                pref = self._normalize_a_prefixed(symbol)
                try:
                    df_adj = ak.stock_zh_a_daily(symbol=pref, adjust=adjust)
                except Exception:
                    df_adj = pd.DataFrame()
                if df_raw is None or df_raw.empty:
                    try:
                        df_raw = ak.stock_zh_a_daily(symbol=pref)
                    except Exception:
                        df_raw = pd.DataFrame()
            if df_adj is None or df_adj.empty:
                return self._standardize(df_adj, market="A", symbol=symbol)
            # 标准化两份数据后按日期对齐，用收盘价比值得到复权因子
            std_adj = self._standardize(df_adj, market="A", symbol=symbol)
            if df_raw is not None and not df_raw.empty:
                std_raw = self._standardize(df_raw, market="A", symbol=symbol)
                m = pd.merge(std_adj[["date","close"]], std_raw[["date","close"]], on="date", suffixes=("_adj","_raw"))
                with pd.option_context('mode.use_inf_as_na', True):
                    m["_af"] = (m["close_adj"].astype(float) / m["close_raw"].astype(float))
                std_adj = std_adj.merge(m[["date","_af"]], on="date", how="left")
                # 统一到单列 adj_factor
                std_adj["adj_factor"] = std_adj["_af"].astype(float)
                std_adj.drop(columns=["_af", "adj_factor_x", "adj_factor_y"], errors="ignore", inplace=True)
            # 若原始数据缺失，保留 NA
            return std_adj
        else:
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily")
            except Exception:
                df = pd.DataFrame()
            # 回退：若为空，尝试带前缀的 daily 接口
            if df is None or df.empty:
                pref = self._normalize_a_prefixed(symbol)
                try:
                    df = ak.stock_zh_a_daily(symbol=pref)
                except Exception:
                    df = pd.DataFrame()
            std = self._standardize(df, market="A", symbol=symbol)
            if not std.empty:
                try:
                    std["adj_factor"] = 1.0
                except Exception:
                    pass
            return std

    def _fetch_daily_hk(self, symbol: str) -> pd.DataFrame:
        sym = self._normalize_hk_symbol(symbol)
        df = pd.DataFrame()
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = "19900101"
            # 优先：带起止日期的接口（部分版本需提供日期才返回）
            try:
                df = ak.stock_hk_daily(symbol=sym, start_date=start, end_date=end)
            except TypeError:
                # 兼容旧版本签名
                df = ak.stock_hk_daily(symbol=sym)
            # 回退：若依然为空，尝试另一接口
            if df is None or df.empty:
                try:
                    df = ak.stock_hk_hist(symbol=sym, period="daily", start_date=start, end_date=end)
                except Exception:
                    pass
        except Exception:
            df = pd.DataFrame()
        # 列中的symbol使用规范化后的
        return self._standardize(df, market="H", symbol=sym)

    def _get_daily(self, market: str, symbol: str, adjust: Optional[str], refresh: bool, expire_days: int) -> pd.DataFrame:
        cache = self._load_cached_daily(market, symbol, adjust)
        cache_path = self._cache_file_daily(market, symbol, adjust)
        need_refresh = refresh or self._is_expired(cache_path, expire_days) or cache.empty
        # 兼容旧缓存：若为 A 股的前/后复权且缓存中缺少或全 NA 的复权因子，则强制刷新
        if not need_refresh and market == "A" and adjust in ("qfq", "hfq"):
            try:
                if ("adj_factor" not in cache.columns) or (cache["adj_factor"].isna().all()):
                    need_refresh = True
            except Exception:
                need_refresh = True
        # 若volume/amount列缺失或全为0/NA，强制刷新一次
        if not need_refresh:
            try:
                vol_ok = ("volume" in cache.columns) and (pd.to_numeric(cache["volume"], errors="coerce").fillna(0).sum() > 0)
                amt_ok = ("amount" in cache.columns) and (pd.to_numeric(cache["amount"], errors="coerce").fillna(0).sum() > 0)
                if not (vol_ok and amt_ok):
                    need_refresh = True
            except Exception:
                need_refresh = True
        if not need_refresh:
            return cache
        # 拉取全量或最新日线
        df_std = self._fetch_daily_a(symbol, adjust) if market == "A" else self._fetch_daily_hk(symbol)
        if not df_std.empty:
            self._save_cached_daily(df_std, market, symbol, adjust)
            return df_std
        # 如果拉取失败则返回已有缓存
        return cache

    def get_a_hist(self, symbol: str, period: PeriodType = "daily", start: Optional[str] = None, end: Optional[str] = None, adjust: Optional[str] = None, *, use_cache: bool = True, refresh: bool = False, expire_days: int = 3) -> pd.DataFrame:
        # 统一以日线为基准 + 本地缓存
        df_d_std = self._get_daily("A", symbol, adjust, refresh if use_cache else True, expire_days)
        if df_d_std.empty:
            return df_d_std
        # 过滤日期
        if start:
            df_d_std = df_d_std[df_d_std["date"] >= pd.to_datetime(start)]
        if end:
            df_d_std = df_d_std[df_d_std["date"] <= pd.to_datetime(end)]
        if period == "daily":
            return df_d_std.reset_index(drop=True)
        return self._resample_ohlcv(df_d_std, period)

    def get_hk_hist(self, symbol: str, period: PeriodType = "daily", start: Optional[str] = None, end: Optional[str] = None, *, use_cache: bool = True, refresh: bool = False, expire_days: int = 3) -> pd.DataFrame:
        # 统一以日线为基准 + 本地缓存
        df_d_std = self._get_daily("H", symbol, None, refresh if use_cache else True, expire_days)
        if df_d_std.empty:
            return df_d_std
        # 过滤日期
        if start:
            df_d_std = df_d_std[df_d_std["date"] >= pd.to_datetime(start)]
        if end:
            df_d_std = df_d_std[df_d_std["date"] <= pd.to_datetime(end)]
        if period == "daily":
            return df_d_std.reset_index(drop=True)
        return self._resample_ohlcv(df_d_std, period)

    def get_hist(self, market: Literal["A","H"], symbol: str, period: PeriodType="daily", start: Optional[str]=None, end: Optional[str]=None, adjust: Optional[str]=None, *, use_cache: bool = True, refresh: bool = False, expire_days: int = 3) -> pd.DataFrame:
        if market == "A":
            return self.get_a_hist(symbol=symbol, period=period, start=start, end=end, adjust=adjust, use_cache=use_cache, refresh=refresh, expire_days=expire_days)
        else:
            return self.get_hk_hist(symbol=symbol, period=period, start=start, end=end, use_cache=use_cache, refresh=refresh, expire_days=expire_days)