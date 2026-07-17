# SQL 从 0 到 1：面向 Data Agent 的数据查询入门

> 这份文档是为理解 DataAgent-Bench 竞赛而写的。不追求成为 DBA，
> 只追求能理解 Agent 生成的 SQL、能调试、能判断对错。

---

## 一、数据库是什么

把数据库想象成一个**超大的 Excel 工作簿**：

```
Excel 工作簿 (.xlsx)
├── Sheet 1: "学生表"     ← 数据库里叫 "表" (table)
│   ├── 列 A: 学号        ← 数据库里叫 "列" (column) 或 "字段" (field)
│   ├── 列 B: 姓名
│   ├── 列 C: GPA
│   └── 第 2~100 行: 数据  ← 数据库里叫 "行" (row) 或 "记录" (record)
│
├── Sheet 2: "课程表"
│   ├── 列 A: 课程号
│   ├── 列 B: 课程名
│   └── 列 C: 学分
│
└── Sheet 3: "选课表"
    ├── 列 A: 学号
    ├── 列 B: 课程号
    └── 列 C: 成绩
```

比赛里的 `.sqlite` / `.db` 文件就是一个数据库，里面有多张表。

---

## 二、SELECT：从表里拿数据

### 最基本的查询

```sql
-- 拿学生表的所有列、所有行
SELECT * FROM students;

-- 只拿姓名和 GPA 两列
SELECT name, gpa FROM students;

-- 只看前 5 行（Preview）
SELECT * FROM students LIMIT 5;
```

`*` = 全部列。`LIMIT` = 只看前 N 行。

### 现实对照

Agent 调用 `read_csv` 做的事，本质上就是：
```sql
SELECT * FROM students LIMIT 20;
```

---

## 三、WHERE：筛选行

类比 Excel 的"筛选"功能。

```sql
-- 只看 GPA > 3.5 的学生
SELECT name, gpa FROM students WHERE gpa > 3.5;

-- 只看计算机专业的学生
SELECT * FROM students WHERE major = 'Computer Science';

-- 多个条件：计算机专业 且 GPA > 3.5
SELECT * FROM students
WHERE major = 'Computer Science'
  AND gpa > 3.5;

-- 计算机或数学专业
SELECT * FROM students
WHERE major = 'Computer Science'
   OR major = 'Mathematics';

-- 名字以 'A' 开头
SELECT * FROM students WHERE name LIKE 'A%';
```

关键符号：
| 符号 | 含义 |
|------|------|
| `=` | 等于 |
| `>` `<` | 大于 / 小于 |
| `>=` `<=` | 大于等于 / 小于等于 |
| `!=` 或 `<>` | 不等于 |
| `LIKE` | 模糊匹配 |
| `%` | 匹配任意字符（LIKE 里用） |
| `AND` | 且 |
| `OR` | 或 |

---

## 四、聚合函数：计算统计值

Excel 里的 SUM()、AVERAGE()、COUNT() 在 SQL 里长这样：

```sql
-- 有多少个学生？
SELECT COUNT(*) FROM students;

-- 所有学生的平均 GPA
SELECT AVG(gpa) FROM students;

-- 最高 GPA
SELECT MAX(gpa) FROM students;

-- 最低 GPA
SELECT MIN(gpa) FROM students;

-- 所有 GPA 的总和
SELECT SUM(gpa) FROM students;
```

### 实战例子

比赛问题："Calculate the mean fare paid by the passengers"

Agent 会生成：
```sql
SELECT AVG(fare) FROM tickets;
-- 或
SELECT AVG(fare) AS mean_fare FROM tickets;
```

`AS` 是给结果列起个别名。`AVG(fare)` 的列名叫 `AVG(fare)` 太长，用 `AS mean_fare` 改成 `mean_fare`。

---

## 五、GROUP BY：分组聚合

Excel 的数据透视表。

```sql
-- 每个专业的平均 GPA
SELECT major, AVG(gpa) AS avg_gpa
FROM students
GROUP BY major;
```

执行过程：
```
原始数据:                    结果:
name    major       gpa      major            avg_gpa
Alice   CS          3.8  →   CS              3.65
Bob     CS          3.5      Math            3.60
Carol   Math        3.7      Physics         3.40
Dave    Math        3.5
Eve     Physics     3.4
```

**口诀**：`GROUP BY` 后面的列 + 聚合函数，才能在 `SELECT` 里出现。普通列不能单独出现。

```sql
-- ❌ 错误：name 既不在 GROUP BY 里，也不是聚合函数
SELECT name, AVG(gpa) FROM students GROUP BY major;

-- ✅ 正确
SELECT major, AVG(gpa) FROM students GROUP BY major;
```

### HAVING：对分组后的结果再筛选

```sql
-- 平均 GPA > 3.5 的专业
SELECT major, AVG(gpa) AS avg_gpa
FROM students
GROUP BY major
HAVING AVG(gpa) > 3.5;
```

`WHERE` 筛的是**原始行**，`HAVING` 筛的是**分组后的结果**。

---

## 六、ORDER BY：排序

```sql
-- 按 GPA 从高到低
SELECT name, gpa FROM students ORDER BY gpa DESC;

-- 按 GPA 从低到高（默认）
SELECT name, gpa FROM students ORDER BY gpa ASC;

-- 先按专业排，同专业内按 GPA 从高到低
SELECT * FROM students ORDER BY major ASC, gpa DESC;
```

`ASC` = 升序（默认），`DESC` = 降序。

---

## 七、JOIN：关联多张表

**这是 SQL 最核心也最容易被用错的部分。** Data Agent 在 medium/hard 任务上翻车，十有八九是 JOIN 写错了。

### 为什么需要 JOIN

现实中数据很少放在一张表里。比如：

```
students 表:                 enrollments 表:
id   name    major           student_id   course_id   grade
1    Alice   CS              1            CS101       A
2    Bob     Math            1            MATH200     B+
3    Carol   CS              2            CS101       A-
                             3            CS101       B
```

问题："列出每个学生选了什么课，拿了什么成绩"

你需要把 `students` 和 `enrollments` 两张表**通过 `id` = `student_id` 这个共同字段拼在一起**。

### INNER JOIN：取交集

```sql
SELECT students.name, enrollments.course_id, enrollments.grade
FROM students
INNER JOIN enrollments ON students.id = enrollments.student_id;
```

结果：
```
name    course_id   grade
Alice   CS101       A
Alice   MATH200     B+
Bob     CS101       A-
Carol   CS101       B
```

**INNER JOIN = 只保留两边都能匹配上的行。** 如果某个学生没选课，他在结果里不会出现。

### LEFT JOIN：保留左表全部

```sql
SELECT students.name, enrollments.course_id, enrollments.grade
FROM students
LEFT JOIN enrollments ON students.id = enrollments.student_id;
```

如果 Dave 在 students 表里但没有选课记录，LEFT JOIN 会保留他，course_id 和 grade 显示 NULL：

```
name    course_id   grade
Alice   CS101       A
Bob     CS101       A-
Carol   CS101       B
Dave    NULL        NULL         ← 没选课也保留了
```

### JOIN 口诀

```
INNER JOIN = 取交集 = 两边都有才保留
LEFT JOIN  = 保左边   = 左边全保留，右边没有的填空
```

---

## 八、子查询：查询里套查询

```sql
-- 找出 GPA 高于平均值的所有学生
SELECT name, gpa
FROM students
WHERE gpa > (SELECT AVG(gpa) FROM students);
```

括号里的 `SELECT AVG(gpa) FROM students` 先执行，得到一个数（比如 3.5），然后外层查询再用这个数去筛选。

---

## 九、WITH (CTE)：给子查询起名

子查询写多了很乱。CTE 让 SQL 像写作文一样分层：

```sql
-- 第一步：算每个专业的平均 GPA
WITH major_avg AS (
    SELECT major, AVG(gpa) AS avg_gpa
    FROM students
    GROUP BY major
)
-- 第二步：找出高于专业平均的学生
SELECT s.name, s.gpa, s.major, m.avg_gpa
FROM students s
JOIN major_avg m ON s.major = m.major
WHERE s.gpa > m.avg_gpa;
```

Agent 在复杂任务里会频繁用 CTE 来组织思路。

---

## 十、Data Agent 中最常见的 SQL 错误

看 Agent 的 trace 时，重点检查这几个：

### 1. JOIN 条件写错
```sql
-- ❌ 用错了关联字段
... ON students.id = enrollments.course_id  -- course_id 不是学生ID！

-- ✅
... ON students.id = enrollments.student_id
```

### 2. 忘了 GROUP BY
```sql
-- ❌ name 不在 GROUP BY 里，SQLite 会报错或返回奇怪结果
SELECT name, AVG(gpa) FROM students;

-- ✅
SELECT AVG(gpa) FROM students;
-- 或
SELECT name, gpa FROM students WHERE ...;
```

### 3. COUNT 用错
```sql
-- COUNT(*) = 数行数
-- COUNT(column) = 数该列非 NULL 的行数
-- COUNT(DISTINCT column) = 数该列有多少个不同的值

-- 有多少个学生选过课？
SELECT COUNT(DISTINCT student_id) FROM enrollments;
-- 不是 COUNT(*)，因为一个学生可能选多门课
```

### 4. 用 = 代替 LIKE
```sql
-- ❌ 只找名字完全等于 'Alice' 的
WHERE name = 'Alice'

-- ✅ 找名字包含 'Alice' 的
WHERE name LIKE '%Alice%'
```

### 5. JOIN 后数据变多（重复行）
```sql
-- 如果 enrollments 表里一个学生有 3 条选课记录
-- JOIN 后这个学生会出现 3 次
-- 此时 COUNT(*) 会翻 3 倍！
-- 解决：用 COUNT(DISTINCT students.id)
```

---

## 十一、如何读懂 Agent 生成的 SQL

在 `trace.json` 里找到 `execute_context_sql` 的调用，`action_input` 里有个 `sql` 字段。按这个顺序读：

1. **先看 FROM** — 从哪些表拿数据？
2. **再看 JOIN** — 关联条件对吗？
3. **再看 WHERE** — 筛选条件合理吗？
4. **再看 GROUP BY / SELECT** — 分组合不合理？选的列对不对？
5. **最后看结果** — observation 里的数据行数、值是否合理

---

## 十二、练习：用竞赛数据练手

```bash
# 找一个有 SQLite 的任务
ls data/public/input/task_22/context/
# 看看里面有没有 .db 或 .sqlite 文件

# 用 sqlite3 命令行打开
sqlite3 data/public/input/task_22/context/*.db

# 在 sqlite3 里试试：
.tables          -- 看有哪些表
.schema          -- 看表结构
SELECT * FROM 表名 LIMIT 5;   -- 看前 5 行
.quit            -- 退出
```

---

## 总结：一张图记住所有 SQL

```
SELECT 列1, 列2, 聚合(列3)    ← 5. 最后决定输出什么
FROM 表1                       ← 1. 从哪张表开始
JOIN 表2 ON 条件               ← 2. 关联其他表
WHERE 条件                     ← 3. 筛选行（聚合前）
GROUP BY 列1                   ← 4. 分组
HAVING 聚合条件                ← 4b. 筛选分组后的结果
ORDER BY 列 DESC               ← 6. 排序
LIMIT 10;                      ← 7. 只要前 N 行
```

**执行顺序**：FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT

理解了这个执行顺序，大部分 SQL 就能读懂了。
