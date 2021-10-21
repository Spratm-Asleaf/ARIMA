function p = PolynomialFit(t, x_, order)
p_x = t;
p_y = x_;
p = polyfit(p_x, p_y, order);
end