import math
class Portfolio:    
    def __init__(self, asset, fiat):
        self.asset = asset          # 可用资产
        self.fiat = fiat           # 可用法币
        self.next_day_available_asset = 0  # 下一个交易日可用的资产

    def valorisation(self, price):
        """计算当前投资组合的总价值"""
        return self.asset * price + self.fiat + self.next_day_available_asset * price

    def position(self, price):
        """计算当前仓位比例（基于可用资产）"""
        total = self.valorisation(price)
        return ((self.asset + self.next_day_available_asset) * price) / total if total != 0 else 0

    def trade_to_position(self, position, price, trading_fees):
        """调整到目标仓位（0-1之间），考虑交易手续费和T+1规则"""
        if position < 0 or position > 1:
            raise ValueError("Position must be between 0 and 1")

        current_value = self.valorisation(price)

        # 计算当前总资产（包括次日到账部分）
        current_total_asset = self.asset + self.next_day_available_asset
        # 目标总资产 = 仓位比例 * 总价值 / 价格
        target_total_asset = (position * current_value) / price
        # 需要交易的资产量 = 目标总资产 - 当前总资产
        asset_trade = target_total_asset - current_total_asset

        # 取整到100的倍数，向零方向
        asset_trade = math.floor(asset_trade / 100) * 100 if asset_trade > 0 else math.ceil(asset_trade / 100) * 100

        if asset_trade > 0:  # 买入操作
            # 计算需要的法币（考虑手续费）
            required_fiat = asset_trade * price * (1 + trading_fees)
            if required_fiat > self.fiat:  # 资金不足时按最大可买量调整
                max_asset = (self.fiat / (price * (1 + trading_fees))) // 100 * 100
                asset_trade = max(max_asset, 0)  # 确保非负
                required_fiat = asset_trade * price * (1 + trading_fees)            
            self.fiat -= required_fiat
            self.next_day_available_asset += asset_trade  # 次日到账

        elif asset_trade < 0:  # 卖出操作
            sell_amount = min(-asset_trade, self.asset)
            self.asset -= sell_amount
            self.fiat += sell_amount * price * (1 - trading_fees)

    def update_day(self):
        """每日结算，更新可用资产"""
        self.asset += self.next_day_available_asset
        self.next_day_available_asset = 0

    def get_portfolio_distribution(self):
        """获取资产分布信息"""
        return {
            "asset": self.asset,
            "fiat": self.fiat,
            "next_day_available_asset": self.next_day_available_asset
        }

    def __str__(self):
        return f"Portfolio(asset={self.asset}, fiat={self.fiat}, next_day_available_asset={self.next_day_available_asset})"

    def describe(self, price):
        print(f"Value: {self.valorisation(price)}, Position: {self.position(price)}")


class TargetPortfolio(Portfolio):
    """直接创建目标仓位的投资组合"""
    def __init__(self, position, value, price):
        if not 0 <= position <= 1:
            raise ValueError("Position must be between 0 and 1")
        target_asset_raw = (position * value) / price
        target_asset = round(target_asset_raw / 100) * 100     
        super().__init__(
            asset= target_asset,
            fiat= value - target_asset * price
        )