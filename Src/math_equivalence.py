def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    
    string = _remove_right_units(string)

    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)

    string = string.replace(" ", "")

    string = _fix_fracs(string)

    try:
        numeric_val = float(string)
        if numeric_val == int(numeric_val):
            string = str(int(numeric_val))
        else:
            string = str(numeric_val)
    except:
        pass

    string = _fix_a_slash_b(string)

    return string

def _eval_latex_numeric(s):
    try:
        return float(s)
    except:
        pass
    import re
    m = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except:
            pass
    m = re.match(r'^-\\frac\{([^}]+)\}\{([^}]+)\}$', s)
    if m:
        try:
            return -float(m.group(1)) / float(m.group(2))
        except:
            pass
    return None

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True
        v1 = _eval_latex_numeric(ss1)
        v2 = _eval_latex_numeric(ss2)
        if v1 is not None and v2 is not None and abs(v1 - v2) < 1e-9:
            return True
        return False
    except:
        return str1 == str2