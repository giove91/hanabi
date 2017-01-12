# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-01-12 10:42
from __future__ import unicode_literals

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Deck',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=1024, validators=[django.core.validators.RegexValidator(regex='^\\d+ (Red|Blue|White|Yellow|Green|Rainbow) \\d+(,\\d+ (Red|Blue|White|Yellow|Green|Rainbow) \\d+)*$')])),
            ],
        ),
        migrations.CreateModel(
            name='Game',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('deck', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='game.Deck')),
            ],
        ),
        migrations.CreateModel(
            name='Team',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('type', models.CharField(choices=[('H', 'Human'), ('A', 'AI')], default='H', max_length=1)),
            ],
        ),
    ]
